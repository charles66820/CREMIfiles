// That Game From Russia
// Copyright (c) 2009 Michaël Larouche <larouche@kde.org>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
#pragma once

#include <vector>

struct SDL_Surface;

class BlockLine;
class Piece;

/**
 * @brief Tetris board
 */
class Board
{
public:
	Board(void);
	~Board(void);

	/**
	 * Reset the board
	 */
	void reset();

	/**
	 * @brief Check if a block is presend at
	 * given coordinates.
	 *
	 * This method will do the mapping between
	 * absolute coordinate and board position.
	 *
	 * @param x Absolute X coordinate
	 * @param y Absolute Y coordinate
	 * @return true if block is present
	 */
	bool blockAt(int x, int y);

	/**
	 * Add block from piece to the board
	 */ 
	void addBlocks(Piece *piece);

	/**
	 * Get the list of full lines.
	 * @return vector of full lines index position
	 */
	std::vector<int> fullLinesPosition() const;

	/**
	 * Remove lines that are full
	 * @return number of lines removed
	 */
	int removeFullLines();

	/**
	 * Render the board
	 */
	void render(SDL_Surface *surface);

private:
	class Private;
	Private *d;
};
