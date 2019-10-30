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
#include "PieceTShape.h"

#include <SDL.h>

PieceTShape::PieceTShape(void)
: Piece(Piece_TShape)
{
}

PieceTShape::~PieceTShape(void)
{
}

Block *PieceTShape::getBlocks(Direction direction)
{
	Block *newBlocks = createBlockArray(BlockPurple);

	int tempX = x();
	int tempY = y();

	newBlocks[0].moveAbsolute(tempX, tempY);

	switch( direction )
	{
		case DirectionDown:
		{
			// Second
			tempX -= BlockWidth;
			newBlocks[1].moveAbsolute(tempX, tempY);

			// Third
			tempX = x() + BlockWidth;
			newBlocks[2].moveAbsolute(tempX, tempY);

			// Fourth
			tempX = x();
			tempY += BlockHeight;
			newBlocks[3].moveAbsolute(tempX, tempY);
			break;
		}
		case DirectionLeft:
		{
			// Second
			tempY -= BlockHeight;
			newBlocks[1].moveAbsolute(tempX, tempY);

			// Third
			tempY = y() + BlockHeight;
			newBlocks[2].moveAbsolute(tempX, tempY);

			// Fourth
			tempX -= BlockWidth;
			tempY = y();
			newBlocks[3].moveAbsolute(tempX, tempY);

			break;
		}
		case DirectionUp:
		{
			// Second
			tempX += BlockWidth;
			newBlocks[1].moveAbsolute(tempX, tempY);

			// Third
			tempX = x() - BlockWidth;
			newBlocks[2].moveAbsolute(tempX, tempY);

			// Fourth
			tempX = x();
			tempY -= BlockHeight;
			newBlocks[3].moveAbsolute(tempX, tempY);

			break;
		}
		case DirectionRight:
		{
			// Second
			tempY += BlockHeight;
			newBlocks[1].moveAbsolute(tempX, tempY);

			// Third
			tempY = y() - BlockHeight;
			newBlocks[2].moveAbsolute(tempX, tempY);

			// Fourth
			tempX = x() + BlockWidth;
			tempY = y();
			newBlocks[3].moveAbsolute(tempX, tempY);
			break;
		}
	}

	return newBlocks;
}
