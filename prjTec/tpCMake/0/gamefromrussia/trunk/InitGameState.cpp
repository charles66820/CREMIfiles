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
#include "InitGameState.h"

#include "PieceFactory.h"

InitGameState::InitGameState(GameStateMachine *machine)
: GameState(machine)
{
}

InitGameState::~InitGameState(void)
{
}

void InitGameState::enter()
{
	GameData *data = this->data();

	// Reset gameplay data
	data->level = data->selectedLevel;
	data->lines = 0;
	data->score = 0;
	data->dropDownDelay = game()->levelDropDown(data->level);

	data->printer.setValue(TextLines, data->lines);
	data->printer.setValue(TextScore, data->score);
	data->printer.setValue(TextLevel, data->level);

	// Remove all previous blocks from board
	data->board.reset();

	// Init next piece, so that game()->nextPiece doesn't
	// swap a NULL pointer
	data->nextPiece = PieceFactory::self()->randomPiece(&data->board);

	// Play the music
	data->player.changeMusic( data->selectedMusic );
}

void InitGameState::execute()
{
	if( !game()->nextPiece() )
	{
		stateMachine()->setNextState(GameOverStateId);
	}
	else
	{
		stateMachine()->setNextState(DropPieceStateId);
	}
}

void InitGameState::exit()
{
}