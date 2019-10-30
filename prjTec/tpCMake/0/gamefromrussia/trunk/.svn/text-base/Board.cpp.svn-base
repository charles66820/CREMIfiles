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
#include "Board.h"

// STL includes
#include <queue>

// Local includes
#include "consts.h"
#include "BlockLine.h"
#include "Piece.h"

#include <SDL.h>

class Board::Private
{
public:
	~Private()
	{
		std::vector<BlockLine*>::iterator it, itEnd = lines.end();
		for(it = lines.begin(); it != itEnd; ++it)
		{
			delete *it;
		}
	}
	std::vector<BlockLine*> lines;
};

Board::Board()
: d(new Private)
{
}

Board::~Board()
{
	delete d;
}

void Board::reset()
{
	delete d;
	d = new Private;
}

void Board::addBlocks(Piece *piece)
{
	for(int i=0; i<TetraSize; i++)
	{
		Block currentBlock = piece->tetra()[i];
		int boardX = (currentBlock.x()-BoardX)/BlockWidth;
		int boardY = (BoardEndY-currentBlock.y())/BlockHeight;

		if( d->lines.size() == 0 )
		{
			d->lines.push_back( new BlockLine );
		}
		if( d->lines.size()-1 < boardY )
		{
			while( d->lines.size()-1 != boardY )
			{
				d->lines.push_back( new BlockLine );
			}
		}

		d->lines.at(boardY)->setBlock(boardX, currentBlock);
	}
}

bool Board::blockAt(int x, int y)
{
	if( d->lines.size() == 0 )
	{
		return false;
	}

	int boardX = (x-BoardX)/BlockWidth;
	int boardY = (BoardEndY-y)/BlockHeight;

	if( d->lines.size()-1 < boardY )
	{
		return false;
	}

	return d->lines[boardY]->hasBlock(boardX);
}

void Board::render(SDL_Surface *surface)
{
	int y=BoardEndY;

	std::vector<BlockLine*>::const_iterator it, itEnd = d->lines.end();
	for(it = d->lines.begin(); it != itEnd; ++it)
	{
		BlockLine *line = *it;
		line->render(surface, y);
		y -= BlockHeight;
	}
}

std::vector<int> Board::fullLinesPosition() const
{
	std::vector<int> linesPosition;

	int y=BoardEndY;

	std::vector<BlockLine*>::const_iterator it, itEnd = d->lines.end();
	for(it = d->lines.begin(); it != itEnd; ++it)
	{
		BlockLine *line = *it;
		if( line->isFull() )
		{
			linesPosition.push_back(y);
		}

		y -= BlockHeight;
	}

	return linesPosition;
}

int Board::removeFullLines()
{
	std::queue<BlockLine*> linesToRemove;

	std::vector<BlockLine*>::iterator it, itEnd = d->lines.end();
	for(it = d->lines.begin(); it != itEnd; ++it)
	{
		BlockLine *line = *it;
		if( line->isFull() )
		{
			linesToRemove.push(line);
		}
	}

	int nbLines = linesToRemove.size();
	while(!linesToRemove.empty())
	{
		BlockLine *line = linesToRemove.front();
		linesToRemove.pop();

		for(int i=0; i<d->lines.size(); i++)
		{
			if( d->lines.at(i) == line )
			{
				d->lines.erase( d->lines.begin()+i );
				delete line;
			}
		}
	}

	return nbLines;
}