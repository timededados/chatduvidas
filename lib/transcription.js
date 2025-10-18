import { toFile } from "openai/uploads";
import { openai } from "./openai.js";
import { TRANSCRIBE_MODEL } from "./constants.js";
import { logSection, logObj } from "./logging.js";

export async function transcribeBase64AudioToText(audioStr, mime = "audio/webm") {
	try {
		logSection("Transcrição de áudio");
		const clean = String(audioStr || "").replace(/^data:.*;base64,/, "");
		logObj("audio_base64_len", clean.length);
		const buf = Buffer.from(clean, "base64");
		logObj("audio_bytes", buf.length);
		const ext = mime.includes("mpeg") ? "mp3"
			: mime.includes("wav") ? "wav"
			: mime.includes("ogg") ? "ogg"
			: mime.includes("m4a") ? "m4a"
			: "webm";
		const filename = `audio.${ext}`;
		const file = await toFile(buf, filename, { type: mime });
		const t0 = Date.now();
		const resp = await openai.audio.transcriptions.create({
			model: TRANSCRIBE_MODEL,
			file,
			language: "pt"
		});
		const ms = Date.now() - t0;
		logObj("transcription_ms", ms);
		const text = (resp && (resp.text || resp.transcript || resp?.results?.[0]?.transcript)) || "";
		logObj("transcription_preview", text.slice(0, 200));
		return text.trim();
	} catch (e) {
		logSection("Transcrição de áudio - erro");
		logObj("error", String(e));
		return "";
	}
}
