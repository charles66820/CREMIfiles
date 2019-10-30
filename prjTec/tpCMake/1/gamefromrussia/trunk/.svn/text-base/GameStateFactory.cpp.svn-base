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
#include "GameStateFactory.h"

// Local includes
#include "GameState.h"
#include "InitGameState.h"
#include "DropPieceState.h"
#include "RemoveLinesState.h"
#include "SetNextPieceState.h"
#include "GameOverState.h"
#include "TitleScreenState.h"
#include "PauseState.h"
#include "SetupState.h"
#include "LinesRemoveAnimationState.h"

GameStateFactory::GameStateFactory()
{
}

GameStateFactory::~GameStateFactory()
{
}

GameState *GameStateFactory::create(GameStates stateId, GameStateMachine *machine)
{
	GameState *newState = 0;
	switch(stateId)
	{
		case InitGameStateId:
			newState = new InitGameState(machine);
			break;
		case DropPieceStateId:
			newState = new DropPieceState(machine);
			break;
		case RemoveLinesStateId:
			newState = new RemoveLinesState(machine);
			break;
		case SetNextPieceStateId:
			newState = new SetNextPieceState(machine);
			break;
		case GameOverStateId:
			newState = new GameOverState(machine);
			break;
		case TitleScreenStateId:
			newState = new TitleScreenState(machine);
			break;
		case PauseStateId:
			newState = new PauseState(machine);
			break;
		case SetupStateId:
			newState = new SetupState(machine);
			break;
		case LinesRemoveAnimationStateId:
			newState = new LinesRemoveAnimationState(machine);
			break;
	}

	return newState;
}