LOCAL_PATH := $(call my-dir)
SDL_PATH := ../SDL

include $(CLEAR_VARS)
LOCAL_MODULE := game
LOCAL_SRC_FILES := game.c game_io.c game_rand.c
LOCAL_CFLAGS += -std=c99 -Wall -O3
LOCAL_EXPORT_CFLAGS := -DGAME=1
include $(BUILD_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := recolor
LOCAL_SRC_FILES := $(SDL_PATH)/src/main/android/SDL_android_main.c recolor_sdl.c main_sdl.c
LOCAL_CFLAGS += -std=c99 -Wall -g -DRECOLOR=2
LOCAL_STATIC_LIBRARIES := game
LOCAL_C_INCLUDES := $(LOCAL_PATH)/$(SDL_PATH)/include
LOCAL_SHARED_LIBRARIES := SDL2 SDL2_ttf SDL2_image
LOCAL_LDLIBS := -lGLESv1_CM -lGLESv2 -llog -rdynamic
include $(BUILD_SHARED_LIBRARY)
