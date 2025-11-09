package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/fatih/color"
	"github.com/philippgille/chromem-go"
)

const (
	ollamaEmbed = "http://127.0.0.1:11434/api/embed"
	ollamaGen   = "http://127.0.0.1:11434/api/generate"
	embModel    = "nomic-embed-text"
	llmModel    = "aratan/eve"
	collName    = "docs"
	dbPath      = "rag_db"
	chunkSize   = 400
	overlapPct  = 0.25
	topK        = 10
	maxChars    = 4000
)

var httpCli = &http.Client{Timeout: 60 * time.Second}

/* ---------- llamadas a Ollama ---------- */

func embedOllama(text string) ([]float32, error) {
	pl := map[string]any{"model": embModel, "input": text}
	b, _ := json.Marshal(pl)
	resp, err := httpCli.Post(ollamaEmbed, "application/json", bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var aux struct {
		Embeddings [][]float64 `json:"embeddings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&aux); err != nil {
		return nil, err
	}
	emb := aux.Embeddings[0]
	out := make([]float32, len(emb))
	for i, v := range emb {
		out[i] = float32(v)
	}
	return out, nil
}

func generateOllama(prompt string) (string, error) {
	pl := map[string]any{
		"model":  llmModel,
		"prompt": prompt,
		"stream": false,
	}
	b, _ := json.Marshal(pl)
	resp, err := httpCli.Post(ollamaGen, "application/json", bytes.NewReader(b))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	var aux struct {
		Response string `json:"response"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&aux); err != nil {
		return "", err
	}
	return aux.Response, nil
}

/* ---------- utilidades ---------- */

func loadDir(dir string) (map[string]string, error) {
	out := make(map[string]string)
	err := filepath.Walk(dir, func(p string, i os.FileInfo, e error) error {
		if e != nil || i.IsDir() || filepath.Ext(p) != ".txt" {
			return e
		}
		b, e := os.ReadFile(p)
		if e != nil {
			return e
		}
		out[p] = string(b)
		return nil
	})
	return out, err
}

func chunkText(text string, size int, overlap float64) []string {
	words := strings.Fields(text)
	step := int(float64(size) * (1 - overlap))
	if step <= 0 {
		step = size
	}
	var chunks []string
	for i := 0; i < len(words); i += step {
		end := i + size
		if end > len(words) {
			end = len(words)
		}
		chunks = append(chunks, strings.Join(words[i:end], " "))
		if end == len(words) {
			break
		}
	}
	return chunks
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

/* ---------- main ---------- */

func main() {
	ctx := context.Background()
	db, err := chromem.NewPersistentDB(dbPath, true)
	if err != nil {
		log.Fatalf("no se pudo abrir/crear la BD: %v", err)
	}

	embedFn := func(_ context.Context, text string) ([]float32, error) {
		return embedOllama(text)
	}

	coll, err := db.GetOrCreateCollection(collName, nil, embedFn)
	if err != nil {
		log.Fatal(err)
	}

	in := bufio.NewReader(os.Stdin)
	color.Cyan("=== RAG-go (chromem-go + Ollama)  PERSISTENTE + COLORES ===")
	for {
		color.Yellow("\n[load|query|update|delete|exit]")
		color.Green("> ")
		line, _ := in.ReadString('\n')
		cmd := strings.TrimSpace(strings.ToLower(line))

		switch cmd {
		case "load":
			docs, err := loadDir("files")
			if err != nil {
				log.Println("error leyendo:", err)
				continue
			}
			totalFragments := 0
			for path, content := range docs {
				chunks := chunkText(content, chunkSize, overlapPct)
				ids := make([]string, len(chunks))
				docs := make([]string, len(chunks))
				for i, ch := range chunks {
					ids[i] = fmt.Sprintf("%s_%d", path, i)
					docs[i] = ch
				}
				if err := coll.Add(ctx, ids, nil, nil, docs); err != nil {
					log.Printf("error insertando %s: %v", path, err)
				} else {
					log.Printf("insertados %d fragmentos de %s", len(docs), path)
					totalFragments += len(docs)
				}
			}
			color.Cyan("Total de fragmentos indexados: %d\n", totalFragments)

		case "query":
			color.Magenta("pregunta: ")
			q, _ := in.ReadString('\n')
			q = strings.TrimSpace(q)
			if q == "" {
				continue
			}
			cnt := coll.Count()
			if cnt == 0 {
				color.Red("Colección vacía: carga ficheros primero (load)")
				continue
			}
			n := topK
			if cnt < n {
				n = cnt
			}
			res, err := coll.Query(ctx, q, n, nil, nil)
			if err != nil {
				log.Println("error query:", err)
				continue
			}

			// Construir contexto hasta maxChars
			var ctxParts []string
			totalChars := 0
			for _, doc := range res {
				if totalChars+len(doc.Content) > maxChars {
					break
				}
				ctxParts = append(ctxParts, doc.Content)
				totalChars += len(doc.Content)
			}
			context := strings.Join(ctxParts, "\n")

			// Prompt que invita a responder SIEMPRE
			prompt := fmt.Sprintf(`Usa el siguiente contexto para responder la pregunta.
Si el contexto no basta, responde con lo que sepas.

Contexto:
%s

Pregunta: %s
Respuesta:`, context, q)

			color.Cyan("DEBUG: prompt (primeras 200 chars): %s...\n", prompt[:min(200, len(prompt))])
			ans, err := generateOllama(prompt)
			if err != nil {
				log.Println("error generando:", err)
				continue
			}
			color.Yellow("DEBUG: respuesta cruda >>>%s<<<\n", ans)
			color.Cyan("\nrespuesta: %s\n", ans)

		case "update":
			color.Yellow("id del fragmento: ")
			id, _ := in.ReadString('\n')
			id = strings.TrimSpace(id)
			color.Yellow("nuevo texto: ")
			txt, _ := in.ReadString('\n')
			txt = strings.TrimSpace(txt)
			if err := coll.Delete(ctx, map[string]string{"id": id}, nil, id); err != nil {
				log.Println("error borrando anterior:", err)
				continue
			}
			if err := coll.Add(ctx, []string{id}, nil, nil, []string{txt}); err != nil {
				log.Println("error insertando nuevo:", err)
				continue
			}
			color.Green("fragmento actualizado\n")

		case "delete":
			color.Yellow("id del fragmento: ")
			id, _ := in.ReadString('\n')
			id = strings.TrimSpace(id)
			if err := coll.Delete(ctx, map[string]string{"id": id}, nil, id); err != nil {
				log.Println("error:", err)
				continue
			}
			color.Green("fragmento borrado\n")

		case "exit":
			color.Red("adiós")
			return

		default:
			color.Red("comando no reconocido")
		}
	}
}
