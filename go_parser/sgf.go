package main

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
)

// Move represents a single move (or setup stone).
type Move struct {
	Player string
	Coords *Coord
}

// SgfGame represents a parsed SGF game.
type SgfGame struct {
	BoardSize     int
	Moves         []Move
	SetupStones   []Move
	Komi          *float64
	Rules         string
	Result        string
	Handicap      int
	InitialPlayer string
}

type sgfNode struct {
	Props map[string][]string
}

type sgfParser struct {
	data []byte
	pos  int
}

func parseSGFFile(path string) (*SgfGame, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	parser := &sgfParser{data: data}
	nodes, err := parser.parseFirstGame()
	if err != nil {
		return nil, err
	}
	if len(nodes) == 0 {
		return nil, fmt.Errorf("SGF file contains no games")
	}

	root := nodes[0]
	if gm := firstProperty(root, "GM"); gm != "" && gm != "1" {
		return nil, fmt.Errorf("not a Go game: GM[%s]", gm)
	}
	boardSize := 19
	if sizeVals, ok := root.Props["SZ"]; ok && len(sizeVals) > 0 {
		boardSize, err = parseBoardSize(sizeVals[0])
		if err != nil {
			return nil, err
		}
	}

	setup, err := parseSetupStones(root, boardSize)
	if err != nil {
		return nil, err
	}

	var komi *float64
	if raw := firstProperty(root, "KM"); raw != "" {
		value, err := strconv.ParseFloat(raw, 64)
		if err != nil || math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, fmt.Errorf("invalid komi: %s", raw)
		}
		komi = &value
	}
	handicap := 0
	if raw := firstProperty(root, "HA"); raw != "" {
		handicap, err = strconv.Atoi(raw)
		if err != nil || handicap < 0 {
			return nil, fmt.Errorf("invalid handicap: %s", raw)
		}
	}
	initialPlayer := strings.ToUpper(firstProperty(root, "PL"))
	if initialPlayer != "" && initialPlayer != "B" && initialPlayer != "W" {
		return nil, fmt.Errorf("invalid initial player: %s", initialPlayer)
	}
	moves := make([]Move, 0, len(nodes))
	for i, node := range nodes {
		if i > 0 {
			for _, property := range []string{"AB", "AW", "AE", "PL"} {
				if _, exists := node.Props[property]; exists {
					return nil, fmt.Errorf("mid-game %s changes are not supported", property)
				}
			}
		}
		move, ok, err := parseMoveNode(node, boardSize)
		if err != nil {
			return nil, err
		}
		if ok {
			moves = append(moves, move)
		}
	}
	if initialPlayer == "" {
		initialPlayer = "B"
		if len(moves) > 0 {
			initialPlayer = moves[0].Player
		} else if handicap >= 2 {
			initialPlayer = "W"
		}
	}

	return &SgfGame{
		BoardSize:     boardSize,
		Moves:         moves,
		SetupStones:   setup,
		Komi:          komi,
		Rules:         firstProperty(root, "RU"),
		Result:        firstProperty(root, "RE"),
		Handicap:      handicap,
		InitialPlayer: initialPlayer,
	}, nil
}

func firstProperty(node sgfNode, key string) string {
	if values := node.Props[key]; len(values) > 0 {
		return strings.TrimSpace(values[0])
	}
	return ""
}

func parseBoardSize(raw string) (int, error) {
	value := strings.TrimSpace(raw)
	if value == "" {
		return 19, nil
	}
	size, err := strconv.Atoi(value)
	// GTP move labels only cover boards up to 19x19.
	if err != nil || size < 1 || size > len(gtpColumns) {
		return 0, fmt.Errorf("invalid board size: %s", raw)
	}
	return size, nil
}

func parseSetupStones(root sgfNode, boardSize int) ([]Move, error) {
	stones := make([]Move, 0)
	if vals, ok := root.Props["AB"]; ok {
		parsed, err := parseSetupValues("B", vals, boardSize)
		if err != nil {
			return nil, err
		}
		stones = append(stones, parsed...)
	}
	if vals, ok := root.Props["AW"]; ok {
		parsed, err := parseSetupValues("W", vals, boardSize)
		if err != nil {
			return nil, err
		}
		stones = append(stones, parsed...)
	}
	return stones, nil
}

func parseSetupValues(player string, values []string, boardSize int) ([]Move, error) {
	stones := make([]Move, 0)
	for _, value := range values {
		coords, err := expandSetupCoords(value, boardSize)
		if err != nil {
			return nil, err
		}
		for _, coord := range coords {
			coordCopy := coord
			stones = append(stones, Move{Player: player, Coords: &coordCopy})
		}
	}
	return stones, nil
}

func expandSetupCoords(value string, boardSize int) ([]Coord, error) {
	value = strings.TrimSpace(strings.ToLower(value))
	if value == "" {
		return nil, fmt.Errorf("invalid setup stone")
	}
	if strings.Contains(value, ":") {
		parts := strings.Split(value, ":")
		if len(parts) != 2 {
			return nil, fmt.Errorf("invalid setup range: %s", value)
		}
		start, err := parseCoord(parts[0], boardSize)
		if err != nil {
			return nil, err
		}
		end, err := parseCoord(parts[1], boardSize)
		if err != nil {
			return nil, err
		}
		return expandCoordRange(start, end), nil
	}
	coord, err := parseCoord(value, boardSize)
	if err != nil {
		return nil, err
	}
	return []Coord{coord}, nil
}

func expandCoordRange(start Coord, end Coord) []Coord {
	minRow := start.Row
	maxRow := end.Row
	if minRow > maxRow {
		minRow, maxRow = maxRow, minRow
	}
	minCol := start.Col
	maxCol := end.Col
	if minCol > maxCol {
		minCol, maxCol = maxCol, minCol
	}
	coords := make([]Coord, 0, (maxRow-minRow+1)*(maxCol-minCol+1))
	for r := minRow; r <= maxRow; r++ {
		for c := minCol; c <= maxCol; c++ {
			coords = append(coords, Coord{Row: r, Col: c})
		}
	}
	return coords
}

func parseMoveNode(node sgfNode, boardSize int) (Move, bool, error) {
	if len(node.Props["B"]) > 1 || len(node.Props["W"]) > 1 || (len(node.Props["B"]) > 0 && len(node.Props["W"]) > 0) {
		return Move{}, false, fmt.Errorf("SGF node must contain at most one move")
	}
	if vals, ok := node.Props["B"]; ok && len(vals) > 0 {
		move, err := parseMove("B", vals[0], boardSize)
		return move, true, err
	}
	if vals, ok := node.Props["W"]; ok && len(vals) > 0 {
		move, err := parseMove("W", vals[0], boardSize)
		return move, true, err
	}
	return Move{}, false, nil
}

func parseMove(player string, value string, boardSize int) (Move, error) {
	value = strings.TrimSpace(strings.ToLower(value))
	if value == "" {
		return Move{Player: player, Coords: nil}, nil
	}
	// FF[4] still allows FF[3]'s [tt] pass on boards up to 19x19 (as sgfmill does).
	if value == "tt" && boardSize <= 19 {
		return Move{Player: player, Coords: nil}, nil
	}
	coord, err := parseCoord(value, boardSize)
	if err != nil {
		return Move{}, err
	}
	return Move{Player: player, Coords: &coord}, nil
}

func parseCoord(value string, boardSize int) (Coord, error) {
	if len(value) != 2 {
		return Coord{}, fmt.Errorf("invalid coordinate: %s", value)
	}
	col := int(value[0] - 'a')
	row := int(value[1] - 'a')
	if col < 0 || row < 0 || col >= boardSize || row >= boardSize {
		return Coord{}, fmt.Errorf("coordinate out of bounds: %s", value)
	}
	return Coord{Row: row, Col: col}, nil
}

func (p *sgfParser) parseFirstGame() ([]sgfNode, error) {
	for !p.eof() && p.peek() != '(' {
		p.pos++
	}
	if p.eof() {
		return nil, fmt.Errorf("SGF file contains no games")
	}
	return p.parseGameTree(true)
}

func (p *sgfParser) parseGameTree(collect bool) ([]sgfNode, error) {
	p.skipWS()
	if p.eof() || p.peek() != '(' {
		return nil, fmt.Errorf("invalid SGF: expected '('")
	}
	p.pos++

	nodes := make([]sgfNode, 0)
	for {
		p.skipWS()
		if p.eof() {
			return nil, fmt.Errorf("invalid SGF: unexpected EOF")
		}
		switch p.peek() {
		case ';':
			node, err := p.parseNode()
			if err != nil {
				return nil, err
			}
			if collect {
				nodes = append(nodes, node)
			}
		default:
			goto doneSequence
		}
	}

doneSequence:
	p.skipWS()
	if !p.eof() && p.peek() == '(' {
		if collect {
			mainNodes, err := p.parseGameTree(true)
			if err != nil {
				return nil, err
			}
			nodes = append(nodes, mainNodes...)
			for {
				p.skipWS()
				if p.eof() || p.peek() != '(' {
					break
				}
				if _, err := p.parseGameTree(false); err != nil {
					return nil, err
				}
			}
		} else {
			for {
				p.skipWS()
				if p.eof() || p.peek() != '(' {
					break
				}
				if _, err := p.parseGameTree(false); err != nil {
					return nil, err
				}
			}
		}
	}

	p.skipWS()
	if p.eof() || p.peek() != ')' {
		return nil, fmt.Errorf("invalid SGF: expected ')'")
	}
	p.pos++
	return nodes, nil
}

func (p *sgfParser) parseNode() (sgfNode, error) {
	if p.eof() || p.peek() != ';' {
		return sgfNode{}, fmt.Errorf("invalid SGF: expected ';'")
	}
	p.pos++
	props := make(map[string][]string)

	for {
		p.skipWS()
		if p.eof() {
			return sgfNode{Props: props}, nil
		}
		ch := p.peek()
		if ch == ';' || ch == '(' || ch == ')' {
			return sgfNode{Props: props}, nil
		}
		ident := p.parsePropIdent()
		if ident == "" {
			return sgfNode{}, fmt.Errorf("invalid SGF: missing property identifier")
		}
		values := make([]string, 0)
		for {
			p.skipWS()
			if p.eof() || p.peek() != '[' {
				break
			}
			value, err := p.parseValue()
			if err != nil {
				return sgfNode{}, err
			}
			values = append(values, value)
		}
		if len(values) == 0 {
			return sgfNode{}, fmt.Errorf("invalid SGF: property %s missing value", ident)
		}
		props[ident] = append(props[ident], values...)
	}
}

func (p *sgfParser) parsePropIdent() string {
	start := p.pos
	for !p.eof() {
		ch := p.peek()
		if ch < 'A' || ch > 'Z' {
			break
		}
		p.pos++
	}
	return string(p.data[start:p.pos])
}

func (p *sgfParser) parseValue() (string, error) {
	if p.eof() || p.peek() != '[' {
		return "", fmt.Errorf("invalid SGF: expected '['")
	}
	p.pos++
	var sb strings.Builder
	for !p.eof() {
		ch := p.next()
		if ch == ']' {
			return sb.String(), nil
		}
		if ch == '\\' {
			if p.eof() {
				break
			}
			next := p.next()
			if next == '\n' || next == '\r' {
				if next == '\r' && !p.eof() && p.peek() == '\n' {
					p.pos++
				}
				continue
			}
			sb.WriteByte(next)
			continue
		}
		sb.WriteByte(ch)
	}
	return "", fmt.Errorf("invalid SGF: unterminated value")
}

func (p *sgfParser) skipWS() {
	for !p.eof() {
		switch p.peek() {
		case ' ', '\t', '\n', '\r':
			p.pos++
		default:
			return
		}
	}
}

func (p *sgfParser) peek() byte {
	return p.data[p.pos]
}

func (p *sgfParser) next() byte {
	ch := p.data[p.pos]
	p.pos++
	return ch
}

func (p *sgfParser) eof() bool {
	return p.pos >= len(p.data)
}
