package main

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
)

const (
	stoneEmpty = 0
	stoneBlack = 1
	stoneWhite = 2
)

var gtpColumns = "ABCDEFGHJKLMNOPQRST"

type Coord struct {
	Row int
	Col int
}

type BoardState struct {
	size        int
	moves       []Move
	promptBuild strings.Builder
	promptBad   bool
	board       []byte
	prevBoard   []byte
	invalid     bool
}

func NewBoardState(size int) *BoardState {
	state := &BoardState{
		size:      size,
		moves:     make([]Move, 0),
		promptBad: false,
		board:     make([]byte, size*size),
		prevBoard: nil,
		invalid:   false,
	}
	state.promptBuild.WriteString("board_size: ")
	state.promptBuild.WriteString(strconv.Itoa(size))
	state.promptBuild.WriteString("\n")
	return state
}

func (b *BoardState) Prompt() (string, error) {
	if b.promptBad {
		return "", fmt.Errorf("invalid prompt state")
	}
	return b.promptBuild.String() + "move:", nil
}

func (b *BoardState) ApplyMove(player string, move string, strict bool) error {
	player = strings.ToUpper(strings.TrimSpace(player))
	if player != "B" && player != "W" {
		return fmt.Errorf("unknown player: %s", player)
	}
	if b.invalid {
		return fmt.Errorf("illegal move for basic rules: prior move invalid")
	}
	coords, err := gtpToCoords(move, b.size)
	if err != nil {
		return err
	}

	if strict {
		return b.applyStrict(player, coords)
	}
	return b.applyBasic(player, coords)
}

func (b *BoardState) applyBasic(player string, coords *Coord) error {
	if b.invalid {
		return fmt.Errorf("illegal move for basic rules: prior move invalid")
	}
	if coords != nil {
		if coords.Row < 0 || coords.Row >= b.size || coords.Col < 0 || coords.Col >= b.size {
			return fmt.Errorf("illegal move for basic rules: out of bounds")
		}
		idx := coords.Row*b.size + coords.Col
		if b.board[idx] != stoneEmpty {
			return fmt.Errorf("illegal move for basic rules: occupied")
		}
	}

	b.appendMove(player, coords)

	if coords == nil {
		b.prevBoard = copyBoard(b.board)
		return nil
	}

	color := stoneBlack
	if player == "W" {
		color = stoneWhite
	}
	next, err := applyStrictMove(b.board, b.size, byte(color), *coords)
	if err != nil {
		b.invalid = true
		return nil
	}
	b.prevBoard = copyBoard(b.board)
	b.board = next
	return nil
}

func (b *BoardState) applyStrict(player string, coords *Coord) error {
	if coords == nil {
		b.appendMove(player, nil)
		b.prevBoard = copyBoard(b.board)
		return nil
	}
	if coords.Row < 0 || coords.Row >= b.size || coords.Col < 0 || coords.Col >= b.size {
		return fmt.Errorf("illegal move for strict rules: out of bounds")
	}
	idx := coords.Row*b.size + coords.Col
	if b.board[idx] != stoneEmpty {
		return fmt.Errorf("illegal move for strict rules: occupied")
	}

	color := stoneBlack
	if player == "W" {
		color = stoneWhite
	}

	next, err := applyStrictMove(b.board, b.size, byte(color), *coords)
	if err != nil {
		return fmt.Errorf("illegal move for strict rules: %w", err)
	}

	if b.prevBoard != nil && boardsEqual(b.prevBoard, next) {
		return fmt.Errorf("illegal move for strict rules: ko")
	}

	b.appendMove(player, coords)
	b.prevBoard = copyBoard(b.board)
	b.board = next
	return nil
}

func (b *BoardState) appendMove(player string, coords *Coord) {
	b.moves = append(b.moves, Move{Player: player, Coords: coords})
	gtp, err := coordsToGTP(coords, b.size)
	if err != nil {
		b.promptBad = true
		return
	}
	b.promptBuild.WriteString(player)
	b.promptBuild.WriteString(": ")
	b.promptBuild.WriteString(gtp)
	b.promptBuild.WriteString("\n")
}

func gtpToCoords(move string, boardSize int) (*Coord, error) {
	move = strings.ToUpper(strings.TrimSpace(move))
	if move == "PASS" {
		return nil, nil
	}
	if len(move) < 2 {
		return nil, fmt.Errorf("invalid GTP move: %s", move)
	}
	colLetter := move[0]
	col := strings.IndexByte(gtpColumns, colLetter)
	if col == -1 {
		return nil, fmt.Errorf("invalid column letter: %c", colLetter)
	}
	row, err := strconv.Atoi(move[1:])
	if err != nil {
		return nil, fmt.Errorf("invalid row number: %s", move)
	}
	if row < 1 || row > boardSize || col < 0 || col >= boardSize {
		return nil, fmt.Errorf("move out of bounds: %s", move)
	}
	return &Coord{Row: boardSize - row, Col: col}, nil
}

func coordsToGTP(coords *Coord, boardSize int) (string, error) {
	if coords == nil {
		return "PASS", nil
	}
	row, col := coords.Row, coords.Col
	if row < 0 || row >= boardSize || col < 0 || col >= boardSize {
		return "", fmt.Errorf("coords out of bounds: (%d, %d)", row, col)
	}
	if col >= len(gtpColumns) {
		return "", fmt.Errorf("invalid column index: %d", col)
	}
	colLetter := gtpColumns[col]
	return fmt.Sprintf("%c%d", colLetter, boardSize-row), nil
}

func applyStrictMove(board []byte, size int, color byte, coords Coord) ([]byte, error) {
	idx := coords.Row*size + coords.Col
	if board[idx] != stoneEmpty {
		return nil, errors.New("move occupied")
	}
	newBoard := copyBoard(board)
	newBoard[idx] = color
	opponent := stoneBlack
	if color == stoneBlack {
		opponent = stoneWhite
	}

	visited := make([]bool, len(newBoard))
	capturedAny := false
	neighbors := neighborIndices(coords.Row, coords.Col, size)
	for _, nIdx := range neighbors {
		if newBoard[nIdx] != byte(opponent) || visited[nIdx] {
			continue
		}
		group, hasLiberty := groupAndLiberties(newBoard, size, nIdx, visited)
		if !hasLiberty {
			capturedAny = true
			for _, g := range group {
				newBoard[g] = stoneEmpty
			}
		}
	}

	selfVisited := make([]bool, len(newBoard))
	_, hasLiberty := groupAndLiberties(newBoard, size, idx, selfVisited)
	if !hasLiberty && !capturedAny {
		return nil, errors.New("move is suicide")
	}
	return newBoard, nil
}

func groupAndLiberties(board []byte, size int, start int, visited []bool) ([]int, bool) {
	color := board[start]
	group := make([]int, 0, size*size)
	stack := []int{start}
	hasLiberty := false
	for len(stack) > 0 {
		idx := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if visited[idx] {
			continue
		}
		visited[idx] = true
		group = append(group, idx)
		row := idx / size
		col := idx % size
		for _, nIdx := range neighborIndices(row, col, size) {
			val := board[nIdx]
			if val == stoneEmpty {
				hasLiberty = true
			} else if val == color && !visited[nIdx] {
				stack = append(stack, nIdx)
			}
		}
	}
	return group, hasLiberty
}

func neighborIndices(row int, col int, size int) []int {
	neighbors := make([]int, 0, 4)
	if row > 0 {
		neighbors = append(neighbors, (row-1)*size+col)
	}
	if row+1 < size {
		neighbors = append(neighbors, (row+1)*size+col)
	}
	if col > 0 {
		neighbors = append(neighbors, row*size+(col-1))
	}
	if col+1 < size {
		neighbors = append(neighbors, row*size+(col+1))
	}
	return neighbors
}

func copyBoard(board []byte) []byte {
	dup := make([]byte, len(board))
	copy(dup, board)
	return dup
}

func boardsEqual(a []byte, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
