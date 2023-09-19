# Streaming Web Server

## Overview

This project is focused on streaming data from a web server to a client using sockets. It captures webcam data on the client and displays it via a Flask web server on the server-side.

**Inspiration**: [Watch this video](https://www.youtube.com/watch?v=JIPbilHxFbI&t=214s)

## Client Setup

**File**: `client-with-webcam.py`

**To Run**:

```bash
python client-with-webcam.py
```

**Features**:

- Captures webcam data
- Exposes a socket on port 8485

## Server Setup

**File**: `webapp-streaming.py`

**To Run**:

```bash
python webapp-streaming.py
```

**Features**:

1. Exposes the URL `http://localhost:5000/video`
2. Connects to the client socket
3. Reads data from the client socket
4. Runs Detectron2 on the image
5. Overlays results on the image
6. Displays the image in a connected browser
