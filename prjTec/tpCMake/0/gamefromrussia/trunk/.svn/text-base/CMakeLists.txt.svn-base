cmake_minimum_required(VERSION 2.6)

project(ThatGameFromRussia)

find_package(SDL REQUIRED)
find_package(SDL_image REQUIRED)
find_package(SDL_mixer REQUIRED)
find_package(SDL_ttf REQUIRED)

if(SDL_FOUND)
	message(STATUS "SDL found at ${SDL_INCLUDE_DIR}")
endif(SDL_FOUND)

if(SDLIMAGE_FOUND)
	message(STATUS "SDL_image found at ${SDLIMAGE_INCLUDE_DIR}")
endif(SDLIMAGE_FOUND)

if(SDLMIXER_FOUND)
	message(STATUS "SDL_mixer found at ${SDLMIXER_INCLUDE_DIR}")
endif(SDLMIXER_FOUND)

if(SDLTTF_FOUND)
	message(STATUS "SDL_ttf found at ${SDLTTF_INCLUDE_DIR}")
endif(SDLTTF_FOUND)

include_directories( ${SDL_INCLUDE_DIR} ${SDLIMAGE_INCLUDE_DIR} ${SDLMIXER_INCLUDE_DIR} ${SDLTTF_INCLUDE_DIR} )

add_definitions(-DDATADIR="${CMAKE_CURRENT_SOURCE_DIR}/data/")

set(gamefromrussia_SRCS
Block.cpp
BlockLine.cpp
BlockSurfaceManager.cpp
Board.cpp
Cursor.cpp
DropPieceState.cpp
Game.cpp
GameOverState.cpp
GameState.cpp
GameStateFactory.cpp
GameStateMachine.cpp
InitGameState.cpp
LinesRemoveAnimationState.cpp
main.cpp
PauseState.cpp
Piece.cpp
PieceFactory.cpp
PieceLongBar.cpp
PieceLShape.cpp
PieceReverseLShape.cpp
PieceReverseSShape.cpp
PieceSquare.cpp
PieceSShape.cpp
PieceTShape.cpp
RemoveLinesState.cpp
SetNextPieceState.cpp
SetupState.cpp
SoundPlayer.cpp
TextPrinter.cpp
TitleScreenState.cpp
)

add_executable(thatgamefromrussia ${gamefromrussia_SRCS})

target_link_libraries(thatgamefromrussia ${SDL_LIBRARY} ${SDLIMAGE_LIBRARY} ${SDLMIXER_LIBRARY} ${SDLTTF_LIBRARY})
