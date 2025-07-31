package chunking

import (
	"crypto/md5"
	"encoding/json"
	"fmt"
	"lcbchat/scraper"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"unicode"
)

// Chunk represents a processed chunk of content ready for embedding
type Chunk struct {
	ID          string                 `json:"id"`
	SourceURL   string                 `json:"source_url"`
	Title       string                 `json:"title"`
	Content     string                 `json:"content"`
	ChunkIndex  int                    `json:"chunk_index"`
	Metadata    map[string]interface{} `json:"metadata"`
	WordCount   int                    `json:"word_count"`
	CharCount   int                    `json:"char_count"`
}

// ChunkingConfig holds configuration for text chunking
type ChunkingConfig struct {
	MaxChunkSize    int     // Maximum characters per chunk
	OverlapSize     int     // Characters to overlap between chunks
	MinChunkSize    int     // Minimum characters per chunk (discard smaller chunks)
	SplitOnSentence bool    // Try to split on sentence boundaries
	SplitOnParagraph bool   // Try to split on paragraph boundaries
}

// PreprocessAndChunk processes scraped content into chunks ready for embedding
func PreprocessAndChunk(config ChunkingConfig, content []scraper.ScrapedContent) ([]Chunk, error) {
	var allChunks []Chunk
	
	for _, item := range content {
		// Preprocess the content
		cleanContent := preprocessText(item.Content)
		
		// Skip if content is too short after preprocessing
		if len(cleanContent) < config.MinChunkSize {
			log.Printf("Skipping %s: content too short after preprocessing (%d chars)", item.URL, len(cleanContent))
			continue
		}
		
		// Create chunks from the preprocessed content
		chunks := createChunks(cleanContent, item, config)
		allChunks = append(allChunks, chunks...)
	}
	
	log.Printf("Created %d chunks from %d content items", len(allChunks), len(content))
	return allChunks, nil
}

// preprocessText cleans and normalizes text content
func preprocessText(content string) string {
	// Remove HTML tags if any remain
	htmlTagRegex := regexp.MustCompile(`<[^>]*>`)
	content = htmlTagRegex.ReplaceAllString(content, " ")
	
	// Remove URLs
	urlRegex := regexp.MustCompile(`https?://[^\s]+`)
	content = urlRegex.ReplaceAllString(content, "")
	
	// Remove email addresses
	emailRegex := regexp.MustCompile(`[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}`)
	content = emailRegex.ReplaceAllString(content, "")
	
	// Normalize whitespace
	whitespaceRegex := regexp.MustCompile(`\s+`)
	content = whitespaceRegex.ReplaceAllString(content, " ")
	
	// Remove excessive punctuation
	punctuationRegex := regexp.MustCompile(`[.]{3,}`)
	content = punctuationRegex.ReplaceAllString(content, "...")
	
	// Remove non-printable characters except newlines and tabs
	content = strings.Map(func(r rune) rune {
		if unicode.IsPrint(r) || r == '\n' || r == '\t' {
			return r
		}
		return -1
	}, content)
	
	// Trim whitespace
	content = strings.TrimSpace(content)
	
	return content
}

// createChunks splits content into overlapping chunks
func createChunks(content string, item scraper.ScrapedContent, config ChunkingConfig) []Chunk {
	var chunks []Chunk
	
	// If content is shorter than max chunk size, return as single chunk
	if len(content) <= config.MaxChunkSize {
		chunk := createChunk(content, item, 0, 0, len(content))
		chunks = append(chunks, chunk)
		return chunks
	}
	
	// Split content into chunks
	chunkIndex := 0
	start := 0
	
	for start < len(content) {
		end := start + config.MaxChunkSize
		
		// Don't go beyond content length
		if end >= len(content) {
			end = len(content)
		} else {
			// Try to find a good breaking point
			end = findBreakPoint(content, start, end, config)
		}
		
		chunkContent := content[start:end]
		
		// Skip chunks that are too small
		if len(strings.TrimSpace(chunkContent)) >= config.MinChunkSize {
			chunk := createChunk(chunkContent, item, chunkIndex, start, end)
			chunks = append(chunks, chunk)
			chunkIndex++
		}
		
		// Move start position for next chunk (with overlap)
		if end >= len(content) {
			break
		}
		start = max(end - config.OverlapSize, 0)
	}
	
	return chunks
}

// findBreakPoint tries to find a good place to break text
func findBreakPoint(content string, start, maxEnd int, config ChunkingConfig) int {
	// If we're at the end of content, return as is
	if maxEnd >= len(content) {
		return len(content)
	}
	
	searchStart := maxEnd - 200 // Look back up to 200 chars for a good break
	if searchStart < start {
		searchStart = start
	}
	
	// Try to break on paragraph (double newline)
	if config.SplitOnParagraph {
		if pos := strings.LastIndex(content[searchStart:maxEnd], "\n\n"); pos != -1 {
			return searchStart + pos
		}
	}
	
	// Try to break on sentence
	if config.SplitOnSentence {
		sentenceEnders := []string{". ", "! ", "? ", ".\n", "!\n", "?\n"}
		bestPos := -1
		
		for _, ender := range sentenceEnders {
			if pos := strings.LastIndex(content[searchStart:maxEnd], ender); pos != -1 {
				actualPos := searchStart + pos + len(ender)
				if actualPos > bestPos {
					bestPos = actualPos
				}
			}
		}
		
		if bestPos != -1 {
			return bestPos
		}
	}
	
	// Try to break on word boundary
	if pos := strings.LastIndex(content[searchStart:maxEnd], " "); pos != -1 {
		return searchStart + pos
	}
	
	// If no good break point found, use maxEnd
	return maxEnd
}

// createChunk creates a Chunk struct from content and metadata
func createChunk(content string, item scraper.ScrapedContent, chunkIndex, start, end int) Chunk {
	content = strings.TrimSpace(content)
	
	// Generate unique ID for chunk
	chunkID := fmt.Sprintf("%s_chunk_%d", item.Hash[:8], chunkIndex)
	
	return Chunk{
		ID:         chunkID,
		SourceURL:  item.URL,
		Title:      item.Title,
		Content:    content,
		ChunkIndex: chunkIndex,
		WordCount:  len(strings.Fields(content)),
		CharCount:  len(content),
		Metadata: map[string]interface{}{
			"content_type": item.ContentType,
			"timestamp":   item.Timestamp,
			"hash":        item.Hash,
			"start_pos":   start,
			"end_pos":     end,
			"total_chunks": 0, // Will be updated later if needed
		},
	}
}

// SaveChunks saves processed chunks to JSON files
func SaveChunks(chunks []Chunk, outDirectory string) error {
	if len(chunks) == 0 {
		return fmt.Errorf("no chunks to save")
	}
	
	// Group chunks by source URL for better organization
	chunksBySource := make(map[string][]Chunk)
	for _, chunk := range chunks {
		chunksBySource[chunk.SourceURL] = append(chunksBySource[chunk.SourceURL], chunk)
	}
	
	// Update total_chunks metadata
	for url, urlChunks := range chunksBySource {
		for i := range urlChunks {
			chunksBySource[url][i].Metadata["total_chunks"] = len(urlChunks)
		}
	}
	
	// Save all chunks to a single JSON file
	allChunksFile := filepath.Join(outDirectory, "chunks_all.json")
	if err := saveChunksToJSON(chunks, allChunksFile); err != nil {
		return fmt.Errorf("failed to save all chunks: %w", err)
	}
	
	// Also save chunks grouped by source
	for url, urlChunks := range chunksBySource {
		// Create a safe filename from URL
		hash := fmt.Sprintf("%x", md5.Sum([]byte(url)))[:8]
		filename := fmt.Sprintf("chunks_%s.json", hash)
		filepath := filepath.Join(outDirectory, filename)
		
		if err := saveChunksToJSON(urlChunks, filepath); err != nil {
			log.Printf("Failed to save chunks for %s: %v", url, err)
		}
	}
	
	log.Printf("Saved %d chunks to %s", len(chunks), outDirectory)
	return nil
}

// saveChunksToJSON saves chunks to a JSON file
func saveChunksToJSON(chunks []Chunk, filepath string) error {
	data, err := json.MarshalIndent(chunks, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal chunks: %w", err)
	}
	
	return os.WriteFile(filepath, data, 0644)
}

// ProcessAndSave combines chunking and saving in one operation
func ProcessAndSave(config ChunkingConfig, content []scraper.ScrapedContent, outputDir string) error {
	chunks, err := PreprocessAndChunk(config, content)
	if err != nil {
		return fmt.Errorf("failed to preprocess and chunk: %w", err)
	}
	
	if err := SaveChunks(chunks, outputDir); err != nil {
		return fmt.Errorf("failed to save chunks: %w", err)
	}
	
	return nil
}

func DefaultChunkingConfig() ChunkingConfig {
	return ChunkingConfig{
		MaxChunkSize:     1000,  // ~200-250 tokens for most models
		OverlapSize:      100,   // 10% overlap
		MinChunkSize:     50,    // Discard very small chunks
		SplitOnSentence:  true,
		SplitOnParagraph: true,
	}
}