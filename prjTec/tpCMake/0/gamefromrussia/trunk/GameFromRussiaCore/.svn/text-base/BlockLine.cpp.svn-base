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
#include "BlockLine.h"

#include <SDL.h>

BlockLine::BlockLine()
{
}

BlockLine::~BlockLine()
{
}

bool BlockLine::isFull() const
{
	bool isFull=true;

	for(int i=0; i<BoardWidth; i++)
	{
		if( m_line[i].color() == BlockEmpty )
		{
			isFull = false;
			break;
		}
	}

	return isFull;
}

bool BlockLine::hasBlock(int index) const
{
	if(index >= BoardWidth)
	{
		return false;
	}

	return m_line[index].color() != BlockEmpty;
}

void BlockLine::setBlock(int index, const Block &block)
{
	if(index >= BoardWidth)
	{
		return;
	}
	m_line[index] = block;
}

void BlockLine::render(SDL_Surface *surface, int y)
{
	SDL_Rect blockPosition;
	blockPosition.x = BoardX;
	blockPosition.y = y;
	blockPosition.w = BlockWidth;
	blockPosition.h = BlockHeight;

	for(int i=0; i<BoardWidth; i++)
	{
		if( m_line[i].color() != BlockEmpty )
		{
			m_line[i].render(surface, &blockPosition);
		}
		blockPosition.x += BlockWidth;
	}
}
