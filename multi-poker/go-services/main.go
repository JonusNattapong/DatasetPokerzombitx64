package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
)

type HandHistory struct {
	ID       string   `json:"id"`
	Players  []string `json:"players"`
	Actions  []Action `json:"actions"`
	Hands    []string `json:"hands"`
	Results  []int    `json:"results"`
}

type Action struct {
	Player string `json:"player"`
	Type   string `json:"type"`
	Amount int    `json:"amount"`
}

type HandProcessor struct {
	mu              sync.RWMutex
	processedHands  map[string]*HandHistory
	processingQueue chan *HandHistory
}

func NewHandProcessor() *HandProcessor {
	return &HandProcessor{
		processedHands:  make(map[string]*HandHistory),
		processingQueue: make(chan *HandHistory, 1000),
	}
}

func (hp *HandProcessor) ProcessHand(hand *HandHistory) {
	hp.processingQueue <- hand
}

func (hp *HandProcessor) startProcessing(workers int) {
	for i := 0; i < workers; i++ {
		go func() {
			for hand := range hp.processingQueue {
				// Process the hand
				hp.mu.Lock()
				hp.processedHands[hand.ID] = hand
				hp.mu.Unlock()
			}
		}()
	}
}

func (hp *HandProcessor) GetProcessedHand(id string) (*HandHistory, bool) {
	hp.mu.RLock()
	defer hp.mu.RUnlock()
	hand, exists := hp.processedHands[id]
	return hand, exists
}

func main() {
	processor := NewHandProcessor()
	processor.startProcessing(4) // Start 4 worker goroutines

	// HTTP endpoints for hand history processing
	http.HandleFunc("/process", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var hand HandHistory
		if err := json.NewDecoder(r.Body).Decode(&hand); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		processor.ProcessHand(&hand)
		w.WriteHeader(http.StatusAccepted)
	})

	http.HandleFunc("/hand/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		id := r.URL.Path[len("/hand/"):]
		hand, exists := processor.GetProcessedHand(id)
		if !exists {
			http.Error(w, "Hand not found", http.StatusNotFound)
			return
		}

		json.NewEncoder(w).Encode(hand)
	})

    http.HandleFunc("/evaluate", processor.evaluateHandler)

	fmt.Println("Starting poker hand processing service on :8080...")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func (hp *HandProcessor) evaluateHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }

    var cards []Card
    if err := json.NewDecoder(r.Body).Decode(&cards); err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }

    rank := evaluateHand(cards)
    response := map[string]int{"rank": rank}
    json.NewEncoder(w).Encode(response)
}

type Card struct {
	Rank string `json:"rank"`
	Suit string `json:"suit"`
}

func evaluateHand(cards []Card) int {
    // Basic hand evaluation (simplified for demonstration)
    ranks := make(map[string]int)
    suits := make(map[string]int)
    for _, card := range cards {
        ranks[card.Rank]++
        suits[card.Suit]++
    }

    // Check for flush
    isFlush := false
    for _, count := range suits {
        if count >= 5 {
            isFlush = true
            break
        }
    }

    // Check for straight (simplified - doesn't handle A-5 straights)
    isStraight := false
    rankValues := "23456789TJQKA"
    straightCount := 0
    for i := 0; i < len(rankValues); i++ {
        if _, exists := ranks[string(rankValues[i])]; exists {
            straightCount++
            if straightCount >= 5 {
                isStraight = true
                break
            }
        } else {
            straightCount = 0
        }
    }
	
    if isFlush && isStraight {
        return 9 // Straight flush
    } else if isFlush {
        return 6 // Flush
    } else if isStraight {
        return 5 // Straight
    } else {
		return 0 // High Card
	}
}
