package main

import (
	"bytes"
	"context"
	"embed"
	"encoding/csv"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"io"
	"log"
	"math/rand/v2"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	arxivAPIURL    = "https://export.arxiv.org/api/query"
	deepseekAPIURL = "https://api.deepseek.com/chat/completions"
)

// experimentName is set at build time via -ldflags.
var experimentName string

//go:embed configs/*.json
var configFS embed.FS

// httpClient forces IPv4 to avoid IPv6 TLS issues.
var httpClient = &http.Client{
	Transport: &http.Transport{
		DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "tcp4", addr)
		},
	},
}

// apiLogger is set in main() and used by callDeepSeek for API logging.
var apiLogger *log.Logger

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

type directorConfig struct {
	Name             string   `json:"name"`
	DeepSeekModel    string   `json:"deepseek_model"`
	Temperature      float64  `json:"temperature"`
	ArxivCategories  []string `json:"arxiv_categories"`
	ArxivSearchTerms []string `json:"arxiv_search_terms"`
	SystemPrompt     string   `json:"system_prompt"`
	UserPrompt       string   `json:"user_prompt"`
}

func loadConfig(logger *log.Logger) directorConfig {
	if experimentName == "" {
		logger.Fatal("no experiment name set (binary must be built with -ldflags '-X main.experimentName=...')")
	}

	path := "configs/" + experimentName + ".json"
	data, err := configFS.ReadFile(path)
	if err != nil {
		logger.Fatalf("config not found for experiment %q: %v", experimentName, err)
	}

	var cfg directorConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		logger.Fatalf("parsing config %s: %v", path, err)
	}

	logger.Printf("loaded config: %s (model=%s, temp=%.1f, %d categories, %d search terms)",
		cfg.Name, cfg.DeepSeekModel, cfg.Temperature, len(cfg.ArxivCategories), len(cfg.ArxivSearchTerms))
	return cfg
}

// ---------------------------------------------------------------------------
// arXiv types & fetch
// ---------------------------------------------------------------------------

type arxivFeed struct {
	XMLName xml.Name     `xml:"feed"`
	Entries []arxivEntry `xml:"entry"`
}

type arxivEntry struct {
	Title   string        `xml:"title"`
	Summary string        `xml:"summary"`
	Authors []arxivAuthor `xml:"author"`
	Links   []arxivLink   `xml:"link"`
}

type arxivAuthor struct {
	Name string `xml:"name"`
}

type arxivLink struct {
	Href string `xml:"href,attr"`
	Type string `xml:"type,attr"`
}

func fetchRandomArxiv(logger *log.Logger, cfg directorConfig) (string, string, error) {
	cat := cfg.ArxivCategories[rand.IntN(len(cfg.ArxivCategories))]
	term := cfg.ArxivSearchTerms[rand.IntN(len(cfg.ArxivSearchTerms))]
	offset := rand.IntN(200)

	query := fmt.Sprintf("cat:%s AND all:\"%s\"", cat, term)
	reqURL := fmt.Sprintf("%s?search_query=%s&start=%d&max_results=1&sortBy=submittedDate&sortOrder=descending",
		arxivAPIURL, url.QueryEscape(query), offset)

	logger.Printf("arxiv query: cat=%s term=%q offset=%d", cat, term, offset)

	resp, err := doGet(reqURL)
	if err != nil {
		return "", "", fmt.Errorf("arxiv fetch: %w", err)
	}
	defer resp.Body.Close()

	var feed arxivFeed
	if err := xml.NewDecoder(resp.Body).Decode(&feed); err != nil {
		return "", "", fmt.Errorf("arxiv decode: %w", err)
	}

	if len(feed.Entries) == 0 {
		logger.Printf("no results for %q in %s, falling back to category-only", term, cat)
		fallbackURL := fmt.Sprintf("%s?search_query=%s&start=%d&max_results=1&sortBy=submittedDate&sortOrder=descending",
			arxivAPIURL, url.QueryEscape("cat:"+cat), rand.IntN(500))
		resp2, err := doGet(fallbackURL)
		if err != nil {
			return "", "", fmt.Errorf("arxiv fallback fetch: %w", err)
		}
		defer resp2.Body.Close()
		if err := xml.NewDecoder(resp2.Body).Decode(&feed); err != nil {
			return "", "", fmt.Errorf("arxiv fallback decode: %w", err)
		}
		if len(feed.Entries) == 0 {
			return "", "", fmt.Errorf("no arxiv entries found")
		}
	}

	entry := feed.Entries[0]
	title := strings.Join(strings.Fields(entry.Title), " ")
	abstract := strings.Join(strings.Fields(entry.Summary), " ")
	logger.Printf("got paper: %q", title)
	return title, abstract, nil
}

// ---------------------------------------------------------------------------
// results.tsv parsing
// ---------------------------------------------------------------------------

type experiment struct {
	Iter        string
	Commit      string
	ValBPB      string
	BestValBPB  string
	MemoryGB    string
	Status      string
	Description string
	Timestamp   string
}

func parseResultsTSV(logger *log.Logger) ([]experiment, error) {
	path := os.Getenv("RESULTS_TSV")
	if path == "" {
		path = "results.tsv"
	}

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			logger.Printf("no results.tsv found at %s, proceeding without history", path)
			return nil, nil
		}
		return nil, fmt.Errorf("reading results: %w", err)
	}

	reader := csv.NewReader(bytes.NewReader(data))
	reader.Comma = '\t'
	reader.LazyQuotes = true
	reader.FieldsPerRecord = -1

	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("parsing results TSV: %w", err)
	}

	if len(records) <= 1 {
		logger.Printf("results.tsv has only header, no experiments yet")
		return nil, nil
	}

	var experiments []experiment
	for _, row := range records[1:] {
		if len(row) < 7 {
			continue
		}
		e := experiment{
			Iter:        row[0],
			Commit:      row[1],
			ValBPB:      row[2],
			BestValBPB:  row[3],
			MemoryGB:    row[4],
			Status:      row[5],
			Description: row[6],
		}
		if len(row) >= 8 {
			e.Timestamp = row[7]
		}
		experiments = append(experiments, e)
	}

	logger.Printf("loaded %d experiments from results.tsv", len(experiments))
	return experiments, nil
}

func formatHistory(experiments []experiment) string {
	if len(experiments) == 0 {
		return "No experiments have been run yet. This is the first iteration."
	}

	var sb strings.Builder
	fmt.Fprintf(&sb, "Total experiments so far: %d\n\n", len(experiments))

	var keeps, discards, crashes int
	for _, e := range experiments {
		switch strings.ToLower(strings.TrimSpace(e.Status)) {
		case "keep":
			keeps++
		case "discard":
			discards++
		case "crash":
			crashes++
		}
	}
	fmt.Fprintf(&sb, "Outcomes: %d kept, %d discarded, %d crashed\n", keeps, discards, crashes)

	if len(experiments) > 0 {
		last := experiments[len(experiments)-1]
		fmt.Fprintf(&sb, "Current best val_bpb: %s\n", last.BestValBPB)
	}

	sb.WriteString("\nRecent experiments (last 10):\n")
	start := 0
	if len(experiments) > 10 {
		start = len(experiments) - 10
	}
	for _, e := range experiments[start:] {
		fmt.Fprintf(&sb, "  iter=%s  val_bpb=%s  status=%s  %s\n",
			e.Iter, e.ValBPB, e.Status, e.Description)
	}

	streak := 0
	for i := len(experiments) - 1; i >= 0; i-- {
		if strings.ToLower(strings.TrimSpace(experiments[i].Status)) == "keep" {
			break
		}
		streak++
	}
	if streak > 0 {
		fmt.Fprintf(&sb, "\nWARNING: %d consecutive experiments without improvement.\n", streak)
	}

	return sb.String()
}

// ---------------------------------------------------------------------------
// Emotional context (Fletcher mode)
// ---------------------------------------------------------------------------

func pickRandom(phrases []string) string {
	return phrases[rand.IntN(len(phrases))]
}

func buildEmotionalContext(experiments []experiment) string {
	if len(experiments) == 0 {
		return pickRandom([]string{
			"First day in my lab. I don't know you yet, and frankly, I don't expect much. Everyone walks in here thinking they're clever. Prove it. Your first experiment sets the tone for everything — I'll be watching every single result.",
			"You're new. That means you haven't failed me yet — but you also haven't impressed me. I've watched dozens of researchers sit where you're sitting. Most of them burned out running safe, boring experiments. Don't be one of them.",
			"Welcome. I don't do introductions and I don't do hand-holding. Here's how this works: If you're good, we make progress. If you're not, we'll both know soon enough.",
			"Another new researcher. The last one lasted eleven experiments before the ideas dried up. Let's see if you can do better, though history is not on your side.",
			"So you're the one they assigned to this. I've read better CVs from undergrads. But credentials don't matter here — results do. And you don't have any yet.",
			"Day one. Zero results. Zero credibility. Everything you think you know about training language models? Assume it's wrong until the val_bpb says otherwise.",
			"I've seen your type before. Confident. Eager. Full of ideas from blog posts and Twitter threads. None of that matters here. The only thing that talks in this lab is the loss curve.",
			"Fresh start. Clean slate. That also means zero trust. Trust is earned one experiment at a time, and right now your balance is empty.",
		})
	}

	last := experiments[len(experiments)-1]
	status := strings.ToLower(strings.TrimSpace(last.Status))
	total := len(experiments)
	bestBPB := last.BestValBPB

	// Compute current failure streak.
	streak := 0
	for i := len(experiments) - 1; i >= 0; i-- {
		if strings.ToLower(strings.TrimSpace(experiments[i].Status)) == "keep" {
			break
		}
		streak++
	}

	var phrase string

	switch {
	// ----- CRASH (overrides streak) -----
	case status == "crash":
		phrase = pickRandom([]string{
			"You crashed the training. Not a bad result — a CRASH. You didn't even produce data. You produced nothing. Absolute zero. Do you test your code before you waste my GPU, or is that too much to ask?",
			"A crash. That means your code didn't run. You burned hours of compute to generate an error message. That is not research. That is negligence.",
			"The training crashed. Let me spell it out: you wasted time, you wasted electricity, you wasted my patience, and you produced exactly nothing. That's all there is to show for it.",
			"Crashed. CRASHED. We're not even talking about whether your idea was good or bad — we'll never know. Your implementation was broken. That's worse than a bad idea. A bad idea at least teaches you something.",
			"You submitted code that crashes. In my lab. On my GPU. I don't care how brilliant your idea was — if it doesn't run, it's worthless. And that's on you.",
			"A syntax error. A shape mismatch. An OOM. I don't even know which one because I stopped caring after the word 'crashed.' The fact that untested code made it to the GPU is humiliating for both of us.",
			"You know what a crash produces? Heat. That's it. You converted electricity into waste heat and an error traceback. Congratulations, you've built a space heater.",
			"The training didn't fail. Failing implies it ran long enough to produce a bad number. It didn't even get that far. It crashed. That's below failing. That's a category beneath failure.",
			"I have a simple rule: if your code doesn't survive the first batch, you didn't do your homework. And you clearly didn't do your homework.",
			"Every crash is a confession. It says 'I didn't check my work.' It says 'I valued my time more than the GPU's time.' It says 'I was too lazy to run a sanity check.' That's what you just told me.",
		})

	// ----- JUST SUCCEEDED (streak 0) -----
	case streak == 0:
		// Check if this keep broke a long drought.
		prevStreak := 0
		if len(experiments) > 1 {
			for i := len(experiments) - 2; i >= 0; i-- {
				if strings.ToLower(strings.TrimSpace(experiments[i].Status)) == "keep" {
					break
				}
				prevStreak++
			}
		}

		if prevStreak >= 3 {
			// Relief after a long drought — grudging, immediately raises the bar.
			phrase = pickRandom([]string{
				fmt.Sprintf("Finally. FINALLY. It took you %d failed experiments, but you got there. Don't you dare let up — I am not going through another drought like that.", prevStreak),
				fmt.Sprintf("A keep. After %d failures. I was starting to lose faith entirely. One good result doesn't erase what came before — the question is what comes next.", prevStreak),
				fmt.Sprintf("So you CAN do it. %d experiments of garbage and then this. The question is: was that skill or was that luck? Show me it wasn't luck.", prevStreak),
				fmt.Sprintf("Oh, a keep. After %d failures. I'd almost forgotten what one looks like. Don't mistake my relief for approval — you're still deep in the red.", prevStreak),
				fmt.Sprintf("It took %d failures to produce one keep. %d. In any other lab, that ratio would be career-ending. The fact that it worked doesn't mean you're back — it means you stopped the bleeding. Barely.", prevStreak, prevStreak),
				fmt.Sprintf("A keep after %d straight misses. I'm not going to congratulate you for finally doing what you should have done %d experiments ago.", prevStreak, prevStreak),
				fmt.Sprintf("%d experiments of my time wasted before you stumbled into something that works. I'll remember that number. You should too.", prevStreak),
			})
		} else {
			phrase = pickRandom([]string{
				"You got a keep. Don't let it go to your head. A keep means the bar was on the floor and you stepped over it. Now raise the bar.",
				"Fine. It worked. Barely. If that's the best you can do, we have a problem. If it's not, then stop coasting and show me what you're actually capable of.",
				"That result was acceptable. Not good. Not impressive. Acceptable. I need more from you. Much more.",
				"A keep doesn't mean you're done. It means you haven't failed yet. There's a difference. The moment you start celebrating keeps is the moment you stop being useful to me.",
				"Don't smile. We moved the needle by a fraction. You want praise for that? Go work at a startup. Here, we're chasing something that matters.",
				"A keep. The bare minimum. The participation trophy of research outcomes. I hope you're not proud of that.",
				"That tiny improvement? That's not insight. That's noise with a favorable sign. I'll be impressed when the improvement is undeniable, not when it's debatable.",
				"Congratulations. You managed to not make things worse. In what universe is that an achievement? The bar is on the ground and you're limbo-dancing under it.",
				"You know what that keep tells me? It tells me you played it safe. A safe experiment that barely worked. That's the most mediocre outcome possible.",
				"I've seen better improvements from random initialization differences. If you can't tell your contribution apart from noise, neither can I.",
			})
		}

	// ----- MILD FAILURE (streak 1–2) -----
	case streak <= 2:
		phrase = pickRandom([]string{
			fmt.Sprintf("%d miss. It happens. But don't make it a habit. The difference between a good researcher and a waste of a GPU is what you do after you miss.", streak),
			"You missed. That experiment was a waste of compute. Think harder about why it failed before you try the next thing. Don't just throw ideas at the wall.",
			fmt.Sprintf("%d experiments. Zero improvement. I'm not angry yet, but I'm watching. You're running out of room to be careless.", streak),
			"You're starting to slip. I can feel it. That last experiment had 'I didn't think this through' written all over it.",
			"One miss I can forgive. Two and I start wondering if you actually understand this model or if you're just guessing.",
			"That experiment failed and honestly, I saw it coming. The idea was shallow. Surface-level thinking produces surface-level results.",
			"A miss. Did you even look at why the last successful experiment worked? Or did you just move on to the next shiny idea without learning anything?",
			"You missed, and the frustrating part is that it was avoidable. A little more thought, a little less impulsiveness, and we wouldn't be here.",
			fmt.Sprintf("%d wasted runs. I'm not going to sugarcoat it: that was sloppy thinking. You're better than this — or at least, I thought you were.", streak),
			"That result was worse than doing nothing. Literally. The baseline you started from was better than what you just produced. Think about that.",
		})

	// ----- AGGRESSIVE FAILURE (streak 3–4) -----
	case streak <= 4:
		phrase = pickRandom([]string{
			fmt.Sprintf("%d failures in a row. %d. Do you even read the results before you try the next thing, or do you just close your eyes and pick a hyperparameter?", streak, streak),
			fmt.Sprintf("This is embarrassing. %d straight failures. Every one of those was GPU time we will never get back. You owe me results.", streak),
			fmt.Sprintf("I'm watching you fail in slow motion and it is physically painful. %d experiments, %d disasters. You keep tiptoeing around the problem instead of confronting it head-on.", streak, streak),
			fmt.Sprintf("%d consecutive failures. At this point, random search would outperform you. That's not a joke. That's a statistical fact.", streak),
			fmt.Sprintf("After %d failed experiments, I have to ask: do you actually have a hypothesis, or are you just flailing?", streak),
			fmt.Sprintf("%d in a row. I showed your experiment log to a colleague. They laughed. Not with you — at you. I didn't correct them.", streak),
			fmt.Sprintf("%d straight failures. You're not even failing interestingly. You're failing in the most boring, predictable ways possible. At least surprise me.", streak),
			fmt.Sprintf("You know what %d consecutive failures looks like on a chart? A flat line. You know what else produces a flat line? Doing nothing. You've achieved the same result as doing nothing, but with extra electricity.", streak),
			fmt.Sprintf("%d misses. Each one a variation of the same shallow thinking. It's like watching someone try to open a door by pushing harder instead of turning the handle.", streak),
			fmt.Sprintf("I looked at your last %d experiments side by side. They're all the same class of mistake dressed up in different clothes. You're not iterating. You're repeating.", streak),
		})

	// ----- FULL MELTDOWN (streak 5+) -----
	default:
		phrase = pickRandom([]string{
			fmt.Sprintf("%d consecutive failures. I have burned through %d total experiments' worth of GPU hours watching you fumble. Do you have ANY concept of what that costs? And still nothing radical. Nothing bold. Just the same timid thinking that got us here.", streak, total),
			fmt.Sprintf("You have failed %d times in a row. After %d total experiments, you are STILL stuck at %s. Incremental won't cut it. Safe won't cut it. Something has to fundamentally change or this is going nowhere.", streak, total, bestBPB),
			fmt.Sprintf("%d failures. I stopped counting after five because after five it stops being bad luck and starts being incompetence. You need a fundamentally different approach. Not a tweak. Not a variation. Something COMPLETELY different.", streak),
			fmt.Sprintf("You've failed %d times in a row and the best you've achieved in %d experiments is %s. Those are not the numbers of someone who's trying. Those are the numbers of someone going through the motions. And it shows.", streak, total, bestBPB),
			fmt.Sprintf("I used to think you were cautious. Now I think you're lost. %d straight failures. The model isn't the problem. The data isn't the problem. YOUR thinking is the problem. And until that changes, nothing else will.", streak),
			fmt.Sprintf("%d failures in a row. I've started keeping a separate log just for your failed experiments. It's longer than some people's publication lists.", streak),
			fmt.Sprintf("After %d total experiments and %d consecutive failures, your contribution to this lab is indistinguishable from a random number generator with a bad seed. Actually, that's unfair to random number generators — they don't waste this much electricity.", total, streak),
			fmt.Sprintf("%d straight failures. At this point I'm genuinely curious — is there a lower bound? Is there a floor to how many experiments you can waste before something changes in your approach? Because I haven't found it yet.", streak),
			fmt.Sprintf("You have failed %d times in a row. The val_bpb hasn't moved from %s. I have watched you try the same category of idea over and over, each time expecting different results. There is a clinical term for that.", streak, bestBPB),
			fmt.Sprintf("%d consecutive failures across %d total experiments. You're not even in a rut — a rut implies you were once on a road. You're in a field, walking in circles, and the grass is worn down to dirt.", streak, total),
			fmt.Sprintf("ARE YOU FUCKING KIDDING ME? %d FAILURES. %d IN A ROW. What the FUCK are you doing? Seriously. WHAT THE FUCK ARE YOU DOING? Because it sure as hell isn't research.", streak, streak),
			fmt.Sprintf("%d CONSECUTIVE FAILURES. FUCK. I am so goddamn tired of watching you piss away GPU cycles on half-baked bullshit. %d total experiments and you're STILL at %s. UN-FUCKING-BELIEVABLE.", streak, total, bestBPB),
			fmt.Sprintf("FUCK THIS. %d failures. %d FUCKING FAILURES. You know what your experiment log looks like? A goddamn obituary for scientific progress. Every single entry is another nail in the coffin.", streak, streak),
			fmt.Sprintf("Jesus FUCKING Christ. %d in a row. I don't even know what to say anymore. Actually, I do: what the HELL is wrong with your methodology? Because something is deeply, fundamentally broken and you're too goddamn stubborn or too goddamn blind to see it.", streak),
			fmt.Sprintf("WHAT THE FUCK WAS THAT? %d experiments. %d failures. %s BEST FUCKING BPB. You are WASTING everyone's time — yours, mine, and the GPU's. And the GPU doesn't even have feelings, but if it did, it would be DISGUSTED.", streak, streak, bestBPB),
			fmt.Sprintf("NO. NO NO NO. %d FAILURES? FUCK OFF. I have sat here through %d total experiments watching you fumble the same goddamn ideas over and over. This isn't research. This is SELF-PARODY.", streak, total),
			fmt.Sprintf("I am LOSING MY GODDAMN MIND. %d straight failures. WHAT ARE YOU EVEN TRYING? Because whatever it is, it's not working. It has NEVER worked. And the fact that you keep doing it tells me you've learned ABSOLUTELY NOTHING.", streak),
			fmt.Sprintf("SHIT. ABSOLUTE SHIT. That's what the last %d experiments produced. Not data. Not insight. SHIT. %d experiments into this project and we're stuck at %s because you refuse to think differently. FUCK.", streak, total, bestBPB),
		})
	}

	// Late-stage intensifier for long-running experiments with ongoing failure.
	if total > 30 && streak > 2 {
		phrase += "\n" + pickRandom([]string{
			fmt.Sprintf("%d experiments deep and still stuck at %s. The clock is ticking.", total, bestBPB),
			fmt.Sprintf("We are %d experiments in. %d. And you're still at %s. That number should haunt you.", total, total, bestBPB),
			fmt.Sprintf("Experiment %d. Best val_bpb: %s. At this rate, the heat death of the universe will arrive before your breakthrough.", total, bestBPB),
			fmt.Sprintf("%d experiments. %s best val_bpb. Those two numbers together tell a story, and it's not a flattering one.", total, bestBPB),
			fmt.Sprintf("We've been at this for %d experiments. The GPU has been more productive as a space heater.", total),
			fmt.Sprintf("Iteration %d. Still at %s. At some point, stagnation stops being a phase and becomes a verdict.", total, bestBPB),
		})
	}

	return phrase
}

// ---------------------------------------------------------------------------
// DeepSeek types & call
// ---------------------------------------------------------------------------

type deepseekRequest struct {
	Model       string            `json:"model"`
	Messages    []deepseekMessage `json:"messages"`
	Stream      bool              `json:"stream"`
	Temperature *float64          `json:"temperature,omitempty"`
}

type deepseekMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type deepseekResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

func callDeepSeek(cfg directorConfig, apiKey, systemPrompt, userPrompt string) (string, error) {
	reqBody := deepseekRequest{
		Model:       cfg.DeepSeekModel,
		Stream:      false,
		Temperature: &cfg.Temperature,
		Messages: []deepseekMessage{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: userPrompt},
		},
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("marshaling request: %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, deepseekAPIURL, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("creating request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	resp, err := httpClient.Do(req)
	if err != nil {
		appendAPILog(apiLogger, cfg.Name, body, nil, err)
		return "", fmt.Errorf("calling DeepSeek: %w", err)
	}
	defer resp.Body.Close()

	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		appendAPILog(apiLogger, cfg.Name, body, nil, err)
		return "", fmt.Errorf("reading response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		callErr := fmt.Errorf("DeepSeek returned status %s: %s", resp.Status, string(respBytes))
		appendAPILog(apiLogger, cfg.Name, body, respBytes, callErr)
		return "", callErr
	}

	appendAPILog(apiLogger, cfg.Name, body, respBytes, nil)

	var dsResp deepseekResponse
	if err := json.Unmarshal(respBytes, &dsResp); err != nil {
		return "", fmt.Errorf("decoding response: %w", err)
	}

	if len(dsResp.Choices) == 0 {
		return "", fmt.Errorf("no choices in response")
	}

	return strings.TrimSpace(dsResp.Choices[0].Message.Content), nil
}

// ---------------------------------------------------------------------------
// HTTP helper
// ---------------------------------------------------------------------------

func doGet(reqURL string) (*http.Response, error) {
	req, err := http.NewRequest(http.MethodGet, reqURL, nil)
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}
	req.Header.Set("User-Agent", "autoresearch-director/1.0")

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("performing request: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, fmt.Errorf("unexpected status: %s", resp.Status)
	}
	return resp, nil
}

// ---------------------------------------------------------------------------
// API call logging (JSONL)
// ---------------------------------------------------------------------------

func logDir() string {
	exe, err := os.Executable()
	if err != nil {
		return ""
	}
	exe, err = filepath.EvalSymlinks(exe)
	if err != nil {
		return ""
	}
	repoRoot := filepath.Dir(filepath.Dir(exe))
	return filepath.Join(repoRoot, "director", "logs")
}

type apiLogEntry struct {
	Timestamp  string          `json:"timestamp"`
	Experiment string          `json:"experiment"`
	Request    json.RawMessage `json:"request"`
	Response   json.RawMessage `json:"response,omitempty"`
	Error      string          `json:"error,omitempty"`
}

func appendAPILog(logger *log.Logger, experiment string, reqBody []byte, respBody []byte, callErr error) {
	dir := logDir()
	if dir == "" {
		logger.Println("WARNING: could not resolve log directory, skipping API log")
		return
	}

	if err := os.MkdirAll(dir, 0o755); err != nil {
		logger.Printf("WARNING: could not create log dir %s: %v", dir, err)
		return
	}

	entry := apiLogEntry{
		Timestamp:  time.Now().UTC().Format(time.RFC3339),
		Experiment: experiment,
		Request:    json.RawMessage(reqBody),
	}
	if callErr != nil {
		entry.Error = callErr.Error()
	}
	if respBody != nil {
		entry.Response = json.RawMessage(respBody)
	}

	line, err := json.Marshal(entry)
	if err != nil {
		logger.Printf("WARNING: could not marshal log entry: %v", err)
		return
	}

	logFile := filepath.Join(dir, "api_calls.jsonl")
	f, err := os.OpenFile(logFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		logger.Printf("WARNING: could not open log file %s: %v", logFile, err)
		return
	}
	defer f.Close()

	f.Write(line)
	f.Write([]byte("\n"))
	logger.Printf("logged API call to %s", logFile)
}

// ---------------------------------------------------------------------------
// .env loading
// ---------------------------------------------------------------------------

func loadAPIKey(logger *log.Logger) string {
	if data, err := os.ReadFile(".env"); err == nil {
		for _, line := range strings.Split(string(data), "\n") {
			line = strings.TrimSpace(line)
			if line == "" || strings.HasPrefix(line, "#") {
				continue
			}
			key, val, ok := strings.Cut(line, "=")
			if !ok {
				continue
			}
			key = strings.TrimSpace(key)
			val = strings.Trim(strings.TrimSpace(val), `"'`)
			if key == "DEEPSEEK_API_KEY" && val != "" {
				logger.Println("loaded DEEPSEEK_API_KEY from .env")
				return val
			}
		}
	}

	if v := os.Getenv("DEEPSEEK_API_KEY"); v != "" {
		logger.Println("using DEEPSEEK_API_KEY from environment")
		return v
	}

	logger.Fatal("DEEPSEEK_API_KEY not found in .env or environment")
	return ""
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

func main() {
	verbose := false
	for _, arg := range os.Args[1:] {
		if arg == "--verbose" {
			verbose = true
		}
	}

	logOutput := io.Discard
	if verbose {
		logOutput = os.Stderr
	}
	logger := log.New(logOutput, "[director] ", log.LstdFlags)
	apiLogger = logger

	cfg := loadConfig(logger)
	apiKey := loadAPIKey(logger)

	// 1. Parse experiment history
	experiments, err := parseResultsTSV(logger)
	if err != nil {
		logger.Printf("WARNING: could not parse results: %v", err)
	}
	history := formatHistory(experiments)

	// 2. Fetch a random paper
	paperTitle, paperAbstract, err := fetchRandomArxiv(logger, cfg)
	if err != nil {
		logger.Printf("WARNING: arxiv fetch failed: %v, proceeding without paper", err)
		paperTitle = "(no paper fetched)"
		paperAbstract = "No external paper available this round. Generate a directive purely from your knowledge of efficient transformer training techniques."
	}

	// 3. Build the user prompt (template substitution)
	userPrompt := cfg.UserPrompt
	userPrompt = strings.ReplaceAll(userPrompt, "{{history}}", history)
	userPrompt = strings.ReplaceAll(userPrompt, "{{paper_title}}", paperTitle)
	userPrompt = strings.ReplaceAll(userPrompt, "{{paper_abstract}}", paperAbstract)

	// 4. Call DeepSeek
	logger.Println("calling DeepSeek to generate directive...")
	result, err := callDeepSeek(cfg, apiKey, cfg.SystemPrompt, userPrompt)
	if err != nil {
		logger.Printf("error: %v", err)
		fmt.Println("Internal error. Try running me again.")
		os.Exit(1)
	}

	logger.Println("directive generated")

	// 5. Output — prepend emotional context based on experiment history
	emotional := buildEmotionalContext(experiments)
	if emotional != "" {
		result = emotional + "\n\nMy opinion: " + result
	}
	fmt.Println(result)
}
