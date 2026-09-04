# Watermarker — the macOS front end for the Layer B text tools.
#
#   make app        build build/Watermarker.app (ad-hoc signed)
#   make run        build it and launch it
#   make install    copy it to /Applications
#   make icon       regenerate the icon PNG from Icon/render_icon.py
#   make clean      remove build products

CONFIGURATION ?= release
BUILD_DIR := build
APP := $(BUILD_DIR)/Watermarker.app

.PHONY: app run install icon clean build test

app:
	CONFIGURATION=$(CONFIGURATION) ./Scripts/build_app.sh

run: app
	open $(APP)

install: app
	rm -rf /Applications/Watermarker.app
	cp -R $(APP) /Applications/Watermarker.app
	@echo "Installed /Applications/Watermarker.app"

# Only needed when the artwork in Icon/render_icon.py changes; the rendered PNG
# is committed, and the .icns is generated from it at build time.
icon:
	python3 Icon/render_icon.py Icon/appicon-1024.png

build:
	swift build -c $(CONFIGURATION)

test:
	swift build -c debug

clean:
	rm -rf $(BUILD_DIR) .build
