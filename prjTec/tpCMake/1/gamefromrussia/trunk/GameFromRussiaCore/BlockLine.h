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

#include "Block.h"
#include "consts.h"

struct SDL_Surface;

class BlockLine
{
public:
	BlockLine(void);
	~BlockLine(void);

	/**
	 * Check if the line is full
	 */
	bool isFull() const;

	/**
	 * Check if the line has block at index
	 * @param index Index
	 */
	bool hasBlock(int index) const;

	/**
	 * Set Block at index
	 * @param index Index
	 * @param block Block to set
	 */
	void setBlock(int index, const Block &block);

	/**
	 * Render the block line
	 * @param surface SDL surface to drawn onto
	 * @param y Y position from Board
	 */
	void render(SDL_Surface *surface, int y);

private:
	Block m_line[BoardWidth];
};
