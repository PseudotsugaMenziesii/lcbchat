package scraper

import (
	"context"
	"crypto/md5"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/gocolly/colly/v2"
	"github.com/gocolly/colly/v2/extensions"
	"github.com/ledongthuc/pdf"
	"github.com/temoto/robotstxt"
)

// ScrapedContent represents a piece of content extracted from the website
type ScrapedContent struct {
	URL         string
	Title       string
	Content     string
	ContentType string
	Timestamp   time.Time
	Hash        string
}

// ScraperConfig holds configuration for the scraper
type ScraperConfig struct {
	StartURL        string
	DomainPrefix    string
	MaxDepth        int
	MaxConcurrency  int
	DelayBetween    time.Duration
	RespectRobots   bool
	UserAgent       string
	OutDir       string
	EnableDebug     bool
}

// WebScraper handles the scraping process
type WebScraper struct {
	config       ScraperConfig
	collector    *colly.Collector
	visited      map[string]bool
	visitedMutex sync.RWMutex
	content      []ScrapedContent
	contentMutex sync.Mutex
	robotsData   *robotstxt.RobotsData
	workPool     chan struct{}
}

// NewWebScraper creates a new web scraper instance
func NewWebScraper(config ScraperConfig) (*WebScraper, error) {
	if err := os.MkdirAll(config.OutDir, 0755); err != nil {
		log.Fatalf("Failed to create output directory: %v", err)
		return &WebScraper{}, err
	}
	
	ws := &WebScraper{
		config:  config,
		visited: make(map[string]bool),
		content: make([]ScrapedContent, 0),
	}

	// Create work pool for concurrency control
	ws.workPool = make(chan struct{}, config.MaxConcurrency)

	// Initialize collector with configuration
	ws.collector = colly.NewCollector(
		colly.MaxDepth(config.MaxDepth),
	)

	// Set user agent
	ws.collector.UserAgent = config.UserAgent

	// Enable debug mode if requested
	// if config.EnableDebug {
	// 	ws.collector.Debugger = &debug.LogDebugger{}
	// }

	// Set up rate limiting
	ws.collector.Limit(&colly.LimitRule{
		DomainGlob:  "*",
		Parallelism: config.MaxConcurrency,
		Delay:       config.DelayBetween,
	})

	// Add random user agent rotation
	extensions.RandomUserAgent(ws.collector)

	// Load robots.txt if respect is enabled
	if config.RespectRobots {
		if err := ws.loadRobotsTxt(); err != nil {
			log.Printf("Warning: Could not load robots.txt: %v", err)
		}
	}

	// Set up collectors
	ws.setupCollectors()

	return ws, nil
}

// loadRobotsTxt fetches and parses robots.txt
func (ws *WebScraper) loadRobotsTxt() error {
	baseURL, err := url.Parse(ws.config.StartURL)
	if err != nil {
		return err
	}

	robotsURL := fmt.Sprintf("%s://%s/robots.txt", baseURL.Scheme, baseURL.Host)
	resp, err := http.Get(robotsURL)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode == 200 {
		ws.robotsData, err = robotstxt.FromResponse(resp)
		return err
	}

	return fmt.Errorf("robots.txt not found (status: %d)", resp.StatusCode)
}

// isAllowedByRobots checks if URL is allowed by robots.txt
func (ws *WebScraper) isAllowedByRobots(urlStr string) bool {
	if ws.robotsData == nil {
		return true
	}
	return ws.robotsData.TestAgent(urlStr, ws.config.UserAgent)
}

// setupCollectors configures the colly collectors
func (ws *WebScraper) setupCollectors() {
	// Handle HTML link discovery
	ws.collector.OnHTML("a[href]", func(e *colly.HTMLElement) {
		link := e.Attr("href")
		absoluteURL := e.Request.AbsoluteURL(link)

		if ws.shouldFollowLink(absoluteURL) {
			log.Printf("Following link: %s", absoluteURL)
			e.Request.Visit(absoluteURL)
		}
	})

	// Handle HTML content extraction
	ws.collector.OnHTML("html", func(e *colly.HTMLElement) {
		log.Printf("Extracting content from: %s", e.Request.URL.String())
		ws.extractHTMLContent(e)
	})

	// Handle PDF and other file downloads
	ws.collector.OnResponse(func(r *colly.Response) {
		contentType := r.Headers.Get("Content-Type")
		if strings.Contains(contentType, "pdf") {
			ws.extractPDFContent(r)
		}
	})

	// Error handling
	ws.collector.OnError(func(r *colly.Response, err error) {
		log.Printf("Error scraping %s: %v", r.Request.URL, err)
	})

	// Request handling
	ws.collector.OnRequest(func(r *colly.Request) {
		log.Printf("Visiting: %s", r.URL.String())
	})
}

	// shouldFollowLink determines if a link should be followed
func (ws *WebScraper) shouldFollowLink(link string) bool {
	log.Printf("Checking link: %s", link)
	
	// Check if it matches our domain prefix
	if !strings.HasPrefix(link, ws.config.DomainPrefix) {
		log.Printf("Link doesn't match domain prefix: %s", link)
		return false
	}

	// Check if already visited
	ws.visitedMutex.RLock()
	visited := ws.visited[link]
	ws.visitedMutex.RUnlock()

	if visited {
		log.Printf("Link already visited: %s", link)
		return false
	}

	// Check robots.txt
	if ws.config.RespectRobots && !ws.isAllowedByRobots(link) {
		log.Printf("Blocked by robots.txt: %s", link)
		return false
	}

	// Mark as visited
	ws.visitedMutex.Lock()
	ws.visited[link] = true
	ws.visitedMutex.Unlock()

	log.Printf("Link approved for following: %s", link)
	return true
}

// extractHTMLContent extracts text content from HTML pages
func (ws *WebScraper) extractHTMLContent(e *colly.HTMLElement) {
	// Extract title
	title := e.ChildText("title")
	if title == "" {
		title = e.ChildText("h1")
	}

	// Extract main content (customize selectors based on your target site)
	contentSelectors := []string{
		"main", "article", ".content", "#content", 
		".post-content", ".entry-content", "body",
	}

	var content string
	for _, selector := range contentSelectors {
		if text := e.ChildText(selector); text != "" {
			content = text
			break
		}
	}

	// Clean up content
	//content = ws.cleanText(content)

	if content != "" {
		ws.addContent(ScrapedContent{
			URL:         e.Request.URL.String(),
			Title:       strings.TrimSpace(title),
			Content:     content,
			ContentType: "text/html",
			Timestamp:   time.Now(),
			Hash:        ws.generateHash(content),
		})
	}
}

// extractPDFContent extracts text from PDF files
func (ws *WebScraper) extractPDFContent(r *colly.Response) {
	// Save PDF temporarily
	tempFile := filepath.Join(ws.config.OutDir, "temp.pdf")
	if err := os.WriteFile(tempFile, r.Body, 0644); err != nil {
		log.Printf("Error saving PDF %s: %v", r.Request.URL, err)
		return
	}
	defer os.Remove(tempFile)

	// Extract text from PDF
	file, reader, err := pdf.Open(tempFile)
	if err != nil {
		log.Printf("Error opening PDF %s: %v", r.Request.URL, err)
		return
	}
	defer file.Close()

	var content strings.Builder
	totalPages := reader.NumPage()

	for pageNum := 1; pageNum <= totalPages; pageNum++ {
		page := reader.Page(pageNum)
		if page.V.IsNull() {
			continue
		}

		text, err := page.GetPlainText(nil)
		if err != nil {
			log.Printf("Error extracting text from page %d of %s: %v", pageNum, r.Request.URL, err)
			continue
		}

		content.WriteString(text)
		content.WriteString("\n")
	}

	cleanedContent := content.String()
	if cleanedContent != "" {
		ws.addContent(ScrapedContent{
			URL:         r.Request.URL.String(),
			Title:       ws.extractTitleFromURL(r.Request.URL.String()),
			Content:     cleanedContent,
			ContentType: "application/pdf",
			Timestamp:   time.Now(),
			Hash:        ws.generateHash(cleanedContent),
		})
	}
}

// extractTitleFromURL extracts a title from URL path
func (ws *WebScraper) extractTitleFromURL(urlStr string) string {
	u, err := url.Parse(urlStr)
	if err != nil {
		return urlStr
	}
	
	path := strings.TrimSuffix(u.Path, "/")
	parts := strings.Split(path, "/")
	if len(parts) > 0 {
		return strings.ReplaceAll(parts[len(parts)-1], "-", " ")
	}
	
	return urlStr
}

// generateHash creates MD5 hash for content deduplication
func (ws *WebScraper) generateHash(content string) string {
	hash := md5.Sum([]byte(content))
	return fmt.Sprintf("%x", hash)
}

// addContent safely adds content to the collection
func (ws *WebScraper) addContent(content ScrapedContent) {
	ws.contentMutex.Lock()
	defer ws.contentMutex.Unlock()

	// Check for duplicates
	for _, existing := range ws.content {
		if existing.Hash == content.Hash {
			log.Printf("Duplicate content found, skipping: %s", content.URL)
			return
		}
	}

	ws.content = append(ws.content, content)
	log.Printf("Added content: %s (%d chars)", content.URL, len(content.Content))
}

// Start begins the scraping process
func (ws *WebScraper) Start(ctx context.Context) error {
	// Start scraping
	log.Printf("Starting scrape from: %s", ws.config.StartURL)
	
	done := make(chan error, 1)
	go func() {
		err := ws.collector.Visit(ws.config.StartURL)
		ws.collector.Wait()
		done <- err
	}()

	// Wait for completion or context cancellation
	select {
	case err := <-done:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
	
}

// GetContent returns all scraped content
func (ws *WebScraper) GetContent() []ScrapedContent {
	ws.contentMutex.Lock()
	defer ws.contentMutex.Unlock()
	
	// Return a copy
	result := make([]ScrapedContent, len(ws.content))
	copy(result, ws.content)
	return result
}

// SaveContent saves scraped content to files
// func (ws *WebScraper) SaveContent() error {
// 	content := ws.GetContent()
	
// 	for i, item := range content {
// 		filename := fmt.Sprintf("content_%d_%s.txt", i, ws.generateHash(item.URL)[:8])
// 		filepath := filepath.Join(ws.config.OutputDir, filename)
		
// 		data := fmt.Sprintf("URL: %s\nTitle: %s\nType: %s\nTimestamp: %s\nHash: %s\n\n%s",
// 			item.URL, item.Title, item.ContentType, item.Timestamp.Format(time.RFC3339), item.Hash, item.Content)
		
// 		if err := os.WriteFile(filepath, []byte(data), 0644); err != nil {
// 			return fmt.Errorf("failed to save content to %s: %w", filepath, err)
// 		}
// 	}
	
// 	log.Printf("Saved %d content items to %s", len(content), ws.config.OutputDir)
// 	return nil
// }