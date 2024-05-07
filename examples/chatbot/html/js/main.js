let websocket_uri = 'ws://' + window.location.host + '/transcription';
// let websocket_audio_uri = 'ws://' + window.location.host + '/audio';

// let websocket_uri = "ws://" + "localhost:8080";

let bufferSize = 4096,
  AudioContext,
  context,
  processor,
  input,
  websocket;
var intervalFunction = null;
var recordingTime = 0;
var server_state = 0;
var you_name = "Undeadpool";

var audioContext = null;
var audioWorkletNode = null;
var audio_state = 0;
var available_transcription_elements = 0;
var available_llm_elements = 0;
var available_audio_elements = 0;
var llm_outputs = [];
var new_transcription_element_state = true;
var audio_sources = [];
var audio_source = null;

initWebSocket();

const zeroPad = (num, places) => String(num).padStart(places, "0");

const generateUUID = () => {
  let dt = new Date().getTime();
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (dt + Math.random() * 16) % 16 | 0;
    dt = Math.floor(dt / 16);
    return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
  });
};

function recording_timer() {
  recordingTime++;
  document.getElementById("recording-time").innerHTML =
    zeroPad(parseInt(recordingTime / 60), 2) +
    ":" +
    zeroPad(parseInt(recordingTime % 60), 2) +
    "s";
}

const start_recording = async () => {
  console.log(audioContext);
  try {
    if (audioContext) {
      await audioContext.resume();

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      if (!audioContext) return;
      console.log(audioContext?.state);

      await audioContext.audioWorklet.addModule("js/audio-processor.js");

      const source = audioContext.createMediaStreamSource(stream);
      audioWorkletNode = new AudioWorkletNode(
        audioContext,
        "audio-stream-processor"
      );

      audioWorkletNode.port.onmessage = (event) => {
        if (server_state != 1) {
          console.log("server is not ready!!");
          return;
        }
        const audioData = event.data;
        if (
          websocket &&
          websocket.readyState === WebSocket.OPEN &&
          audio_state == 0
        ) {
          websocket.send(audioData.buffer);
          console.log("send data");
        }
      };

      source.connect(audioWorkletNode);
    }
  } catch (e) {
    console.log("Error", e);
  }
};

const handleStartRecording = async () => {
  audio_state = 0;
  start_recording();
};

const startRecording = async () => {
  document.getElementById("instructions-text").style.display = "none";
  document.getElementById("control-container").style.backgroundColor = "white";

  AudioContext = window.AudioContext || window.webkitAudioContext;
  audioContext = new AudioContext({
    latencyHint: "interactive",
    sampleRate: 16000,
  });

  document.getElementById("recording-stop-btn").style.display = "block";
  document.getElementById("recording-dot").style.display = "block";
  document.getElementById("recording-line").style.display = "block";
  document.getElementById("recording-time").style.display = "block";

  intervalFunction = setInterval(recording_timer, 1000);

  await handleStartRecording();
};

function stopRecording() {
  audio_state = 1;
  clearInterval(intervalFunction);
}

function initWebSocket() {
  websocket = new WebSocket(websocket_uri);
  websocket.binaryType = "arraybuffer";

  console.log("Websocket created.");

  websocket.onopen = function () {
    console.log("Connected to server.");

    websocket.send(
      JSON.stringify({
        uid: generateUUID(),
        multilingual: false,
        language: "en",
        task: "transcribe",
      })
    );
  };

  websocket.onclose = function (e) {
    console.log("Connection closed (" + e.code + ").");
  };

  websocket.onmessage = function (e) {
    var data = JSON.parse(e.data);

    if ("message" in data) {
      if (data["message"] == "SERVER_READY") {
        server_state = 1;
      }
    } else if ("segments" in data) {
      if (new_transcription_element_state) {
        available_transcription_elements = available_transcription_elements + 1;

        var img_src = "udp.png";

        new_transcription_element(you_name, img_src);
        new_text_element(
          "<p>" + data["segments"][0].text + "</p>",
          "transcription-" + available_transcription_elements
        );
        new_transcription_element_state = false;
      }
      document.getElementById(
        "transcription-" + available_transcription_elements
      ).innerHTML = "<p>" + data["segments"][0].text + "</p>";

      if (data["eos"] == true) {
        new_transcription_element_state = true;
      }
    }

    window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
  };
}

function new_transcription_element(speaker_name, speaker_avatar) {
  var avatar_container = document.createElement("div");
  avatar_container.className = "avatar-container";

  var avatar_img = document.createElement("div");
  avatar_img.innerHTML =
    "<img class='avatar' src='img/" + speaker_avatar + "' >";

  var avatar_name = document.createElement("div");
  avatar_name.className = "avatar-name";
  avatar_name.innerHTML = speaker_name;

  var dummy_element = document.createElement("div");

  avatar_container.appendChild(avatar_img);
  avatar_container.appendChild(avatar_name);
  avatar_container.appendChild(dummy_element);

  document.getElementById("main-wrapper").appendChild(avatar_container);
}

function new_text_element(text, id) {
  var text_container = document.createElement("div");
  text_container.className = "text-container";
  text_container.style.maxWidth = "500px";

  var text_element = document.createElement("div");
  text_element.id = id;
  text_element.innerHTML = "<p>" + text + "</p>";

  var dummy_element = document.createElement("div");

  text_container.appendChild(text_element);
  text_container.appendChild(dummy_element);

  document.getElementById("main-wrapper").appendChild(text_container);
}

function new_transcription_time_element(time) {
  var text_container = document.createElement("div");
  text_container.className = "transcription-timing-container";
  text_container.style.maxWidth = "500px";

  var text_element = document.createElement("div");
  text_element.innerHTML =
    "<span>ThonburianWhisper - Transcription time: " + time + "ms</span>";

  var dummy_element = document.createElement("div");

  text_container.appendChild(text_element);
  text_container.appendChild(dummy_element);

  document.getElementById("main-wrapper").appendChild(text_container);
}

document.addEventListener(
  "DOMContentLoaded",
  function () {
    const queryString = window.location.search;
    const urlParams = new URLSearchParams(queryString);
    if (urlParams.has("name")) {
      you_name = urlParams.get("name");
    }
  },
  false
);
