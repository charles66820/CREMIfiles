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
#include "PieceLongBar.h"

#include <SDL.h>

PieceLongBar::PieceLongBar()
: Piece(Piece_LongBar)
{
}

PieceLongBar::~PieceLongBar()
{
}

Block *PieceLongBar::getBlocks(Direction direction)
{
	Block *newBlocks = createBlockArray(BlockCyan);

	int tempX = x();
	int tempY = y();

	// Vertical drawing
	if( direction == DirectionLeft || direction == DirectionRight )
	{
		int i=0;
		for(; i<=TetraSize/2; i++)
		{
			newBlocks[i].moveAbsolute(tempX, tempY);
			tempY -= BlockHeight;
		}
		tempY = y();
		for(; i<TetraSize; i++)
		{
			tempY += BlockHeight;
			newBlocks[i].moveAbsolute(tempX, tempY);
		}
	}
	// Horizontal drawing
	else if( direction == DirectionUp || direction == DirectionDown )
	{
		int i=0;
		for(; i<=TetraSize/2; i++)
		{
			newBlocks[i].moveAbsolute(tempX, tempY);
			tempX -= BlockWidth;
		}
		tempX = x();
		for(; i<TetraSize; i++)
		{
			tempX += BlockWidth;
			newBlocks[i].moveAbsolute(tempX, tempY);
		}
	}

	return newBlocks;
}
