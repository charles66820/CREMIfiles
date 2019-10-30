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

#include "consts.h"
#include "enums.h"
#include "Block.h"

struct SDL_Surface;
struct SDL_Rect;

class Board;

class Piece
{
public:
	virtual ~Piece(void);

	/**
	 * Call to init the class
	 * Here because init() can't be called from
	 * c-tor because init() call a pure virtual method
	 * @param board Board instance
	 */
	void init(Board *board);

	PieceType type() const;

	int x() const;
	int y() const;
	Direction direction() const;

	/**
	 * Step left and detect collision
	 */
	void stepLeft();

	/**
	 * Step right and detect collision
	 */
	void stepRight();

	/**
	 * Drop down, check collection
	 * Return true when the drop has touched something
	 */
	bool dropDown();

	/**
	 * @brief Move the piece in an absolute way
	 *
	 * Does not do any collision detection.
	 */
	void move(int deltaX, int deltaY);

	void render(SDL_Surface *surface);

	void rotate();

	bool tryMove(int deltaX, int deltaY);

	/**
	 * Get tetra array
	 */
	const Block *tetra() const;

protected:
	Piece(PieceType type);

	virtual Block *getBlocks(Direction direction)=0;

	/**
	 * Utility method to create block array
	 */
	Block *createBlockArray(BlockColor color);

private:
	bool tryMove(int deltaX, int deltaY, Block *block);
	void moveBlocks(int deltaX, int deltaY);
	bool isOutsideBoard(const Block &block, int deltaX, int deltaY) const;
	bool hasBlockingBlock(const Block &block, int deltaX, int deltaY);
	
private:
	int m_x;
	int m_y;
	PieceType m_type;
	Direction m_direction;
	Block *m_tetra;
	Board *m_board;
};
