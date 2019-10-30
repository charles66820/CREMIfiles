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

// Local includes
#include "enums.h"
#include "GameStateMachine.h"
#include "Game.h"
#include "GameData.h"

class GameState
{
public:
	virtual ~GameState();

	virtual void enter()=0;
	virtual void execute()=0;
	virtual void exit()=0;

protected:
	GameState(GameStateMachine *machine);

	Game *game() const;

	GameData* data() const;

	GameStateMachine *stateMachine() const;

private:
	GameStateMachine *m_stateMachine;
};

inline Game *GameState::game() const
{
	return stateMachine()->game();
}

inline GameData *GameState::data() const
{
	return stateMachine()->data();
}

inline GameStateMachine *GameState::stateMachine() const
{
	return m_stateMachine;
}
