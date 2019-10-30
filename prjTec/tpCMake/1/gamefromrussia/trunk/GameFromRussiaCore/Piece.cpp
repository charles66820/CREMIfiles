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
#include "Piece.h"

// Local includes
#include "Board.h"
#include "Block.h"

Piece::Piece(PieceType type)
: m_type(type), m_x(0), m_y(0), m_direction(DirectionDown)
{
}

Piece::~Piece(void)
{
	delete[] m_tetra;
}

PieceType Piece::type() const
{
	return m_type;
}

void Piece::init(Board *board)
{
	m_board = board;
	m_tetra = getBlocks( m_direction );
}

void Piece::stepLeft()
{
	if( tryMove(-BlockWidth, 0) )
	{
		m_x -= BlockWidth;
		moveBlocks(-BlockWidth, 0);
	}
}

void Piece::stepRight()
{
	if( tryMove(BlockWidth, 0) )
	{
		m_x += BlockWidth;
		moveBlocks(BlockWidth, 0);
	}
}

bool Piece::dropDown()
{
	if( tryMove(0, BlockHeight) )
	{
		m_y += BlockHeight;
		moveBlocks(0, BlockHeight);
	}
	else
	{
		return true;
	}

	return false;
}

bool Piece::tryMove(int deltaX, int deltaY)
{
	return tryMove(deltaX, deltaY, m_tetra);
}

bool Piece::tryMove(int deltaX, int deltaY, Block *block)
{
	bool canMove=true;

	for(int i=0; i<TetraSize; i++)
	{
		if( isOutsideBoard(block[i], deltaX, deltaY) ||
			hasBlockingBlock(block[i], deltaX, deltaY) )
		{
			canMove=false;
			break;
		}
	}

	return canMove;
}

bool Piece::isOutsideBoard(const Block &block, int deltaX, int deltaY) const
{
	return (block.x() + deltaX) >= BoardEndX ||
		   (block.x() + deltaX) < BoardX ||
		   (block.y() + deltaY) < BoardY ||
		   (block.y() + deltaY) >= BoardEndY;
}

bool Piece::hasBlockingBlock(const Block &block, int deltaX, int deltaY)
{
	return m_board->blockAt(block.x() + deltaX, block.y() + deltaY);
}

void Piece::move(int x, int y)
{
	m_x = x;
	m_y = y;

	delete[] m_tetra;
	m_tetra = getBlocks( direction() );
}

int Piece::x() const
{
	return m_x;
}

int Piece::y() const
{
	return m_y;
}

Direction Piece::direction() const
{
	return m_direction;
}

void Piece::rotate()
{
	Direction newDirection;
	// Rotation is clockwise
	switch(m_direction)
	{
		case DirectionUp:
			newDirection = DirectionRight;
			break;
		case DirectionDown:
			newDirection = DirectionLeft;
			break;
		case DirectionLeft:
			newDirection = DirectionUp;
			break;
		case DirectionRight:
			newDirection = DirectionDown;
			break;
	}

	if( type() == Piece_Square )
	{
		m_direction = newDirection;
		return;
	}

	Block *rotatedBlocks = getBlocks(newDirection);
	if( tryMove(0, 0, rotatedBlocks) )
	{
		delete[] m_tetra;
		m_tetra = rotatedBlocks;
		m_direction = newDirection;
	}
	else
	{
		delete[] rotatedBlocks;
	}
}

void Piece::render(SDL_Surface *surface)
{
	for(int i=0; i<TetraSize; i++)
	{
		m_tetra[i].render(surface);
	}
}

Block *Piece::createBlockArray(BlockColor color)
{
	Block *newBlocks = new Block[4];
	for(int i=0; i<TetraSize; i++)
	{
		newBlocks[i] = Block(color);
	}

	return newBlocks;
}

void Piece::moveBlocks(int deltaX, int deltaY)
{
	for(int i=0; i<TetraSize; i++)
	{
		m_tetra[i].move(deltaX, deltaY);
	}
}

const Block *Piece::tetra() const
{
	return m_tetra;
}