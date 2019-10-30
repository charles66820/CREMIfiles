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
#include "SoundPlayer.h"

// STL includes
#include <iostream>
#include <map>

// SDL includes
#include <SDL.h>
#include <SDL_mixer.h>

// Local includes
#include "consts.h"

using namespace std;

class SoundPlayer::Private
{
public:
	Private()
		: currentMusic(0), hasSoundDevice(true)
	{
		soundCache[SoundDrop] = 0;
		soundCache[SoundMove] = 0;
		soundCache[SoundRotate] = 0;
		soundCache[SoundLineRemove] = 0;
		soundCache[SoundTetris] = 0;
	}
	~Private()
	{
		if( currentMusic )
		{
			Mix_FreeMusic(currentMusic);
		}
	}

	bool HasNoSound() const
	{
		return hasSoundDevice == false;
	}

	Mix_Music *currentMusic;
	map<SoundEffect, Mix_Chunk*> soundCache;
	bool hasSoundDevice;
};

SoundPlayer::SoundPlayer(void)
: d(new Private)
{
}

SoundPlayer::~SoundPlayer(void)
{
	delete d;
	Mix_CloseAudio();
}

void SoundPlayer::init()
{
	if( Mix_OpenAudio(SampleRate, AUDIO_S16SYS, AudioChannels, AudioBufferSize) != 0 )
	{
		cerr << "Unable to initialize audio: " << Mix_GetError() << endl;
		d->hasSoundDevice = false;
	}
}

void SoundPlayer::changeMusic(Music music)
{
	if( d->HasNoSound() )
	{
		return;
	}

	string musicPath;
	switch(music)
	{
		case MusicTitleScreen:
			musicPath = string(DATADIR"MusicTitleScreen.ogg");
			break;
		case MusicGameOver:
			musicPath = string(DATADIR"MusicGameOver.ogg");
			break;
		case MusicNewSchool:
			musicPath = string(DATADIR"MusicNewSchool.ogg");
			break;
		case MusicOldSchool:
			musicPath = string(DATADIR"MusicOldSchool.ogg");
			break;
		case MusicVanHalen:
			musicPath = string(DATADIR"MusicVanHalen.ogg");
			break;
	}

	// Stop current music
	if( d->currentMusic )
	{
		Mix_HaltMusic();
		Mix_FreeMusic(d->currentMusic);
	}

	if( music != MusicNoMusic )
	{
		// Load new one and loop
		d->currentMusic = Mix_LoadMUS(musicPath.c_str());
		if( d->currentMusic )
		{
			// Play music, and play game over music just once
			// The others are looped
			Mix_PlayMusic(d->currentMusic, (music == MusicGameOver) ? 0 : -1 );
		}
	}
	else
	{
		d->currentMusic = 0;
	}
}

void SoundPlayer::pauseMusic()
{
	if( d->currentMusic )
	{
		Mix_PauseMusic();
	}
}

void SoundPlayer::resumeMusic()
{
	Mix_ResumeMusic();
}

void SoundPlayer::stopMusic()
{
	Mix_HaltMusic();
}

void SoundPlayer::playSound(SoundEffect sound)
{
	// Load sound if is not loaded
	if( !d->soundCache[sound] )
	{
		Mix_Chunk *newSound = 0;
		switch(sound)
		{
			case SoundDrop:
				newSound = Mix_LoadWAV(DATADIR"SoundDrop.wav");
				break;
			case SoundMove:
				newSound = Mix_LoadWAV(DATADIR"SoundMove.wav");
				break;
			case SoundRotate:
				newSound = Mix_LoadWAV(DATADIR"SoundRotate.wav");
				break;
			case SoundLineRemove:
				newSound = Mix_LoadWAV(DATADIR"SoundLineRemove.wav");
				break;
			case SoundTetris:
				newSound = Mix_LoadWAV(DATADIR"SoundTetris.wav");
				break;
		}
		d->soundCache[sound] = newSound;
	}

	// Play the sound
	int playChannel = Mix_PlayChannel(-1, d->soundCache[sound], 0);
	if( playChannel == -1 )
	{
		cerr << "Can't play sound " << (int)sound << "." << endl;
	}
}