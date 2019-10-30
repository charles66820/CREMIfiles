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
#include "PieceFactory.h"

// STL includes
#include <ctime>
#include <cstdlib>

// Local includes
#include "Board.h"
#include "PieceLongBar.h"
#include "PieceLShape.h"
#include "PieceReverseLShape.h"
#include "PieceReverseSShape.h"
#include "PieceSquare.h"
#include "PieceSShape.h"
#include "PieceTShape.h"

PieceFactory PieceFactory::s_self;

PieceFactory *PieceFactory::self()
{
	return &s_self;
}

PieceFactory::PieceFactory()
{
	srand(time(NULL));
}

PieceFactory::~PieceFactory()
{
}

Piece *PieceFactory::create(PieceType type, Board *board)
{
	Piece *newPiece = 0;

	switch(type)
	{
		case Piece_Square:
			newPiece = new PieceSquare;
			break;
		case Piece_LongBar:
			newPiece = new PieceLongBar;
			break;
		case Piece_LShape:
			newPiece = new PieceLShape;
			break;
		case Piece_ReverseLShape:
			newPiece = new PieceReverseLShape;
			break;
		case Piece_SShape:
			newPiece = new PieceSShape;
			break;
		case Piece_ReverseSShape:
			newPiece = new PieceReverseSShape;
			break;
		case Piece_TShape:
			newPiece = new PieceTShape;
			break;
	}

	if( newPiece )
	{
		newPiece->init(board);
	}

	return newPiece;
}

Piece *PieceFactory::randomPiece(Board *board)
{
	// If pieceBag is empty, fill it with the next pieces
	if( pieceBag.empty() )
	{
		// Fill bag with Piece_Size distinct random pieces
		while( pieceBag.size() != Piece_Size )
		{
			pieceBag.insert( static_cast<PieceType>(rand() % Piece_Size) );
		}
	}

	// Get next piece and remove it from bag
	PieceType type = *pieceBag.begin();
	pieceBag.erase( pieceBag.begin() );

	return create(type, board);
}
