import express from "express";
import cors from "cors";
import { InferenceClient } from "@huggingface/inference";

const app = express();

app.use(cors());
app.use(express.json({ limit: "5mb" }));

app.get("/", (req, res) => {
  res.send("OKZ AI Video Maker Backend is running!");
});

app.post("/api/generate", async (req, res) => {
  try {
    const { prompt, mode, imageBase64, mimeType } = req.body;

    if (!process.env.HF_TOKEN) {
      return res.status(500).json({ error: "HF_TOKEN is missing" });
    }

    if (!prompt || prompt.length > 1500) {
      return res.status(400).json({ error: "Invalid prompt" });
    }

    const client = new InferenceClient(process.env.HF_TOKEN);
    const model = process.env.HF_VIDEO_MODEL || "Wan-AI/Wan2.2-TI2V-5B";

    let video;

    if (mode === "text") {
      video = await client.textToVideo({ model, inputs: prompt });
    } else if (mode === "image") {
      if (!imageBase64) {
        return res.status(400).json({ error: "Image required" });
      }

      const image = Buffer.from(imageBase64, "base64");

      video = await client.imageToVideo({
        model,
        inputs: new Blob([image], { type: mimeType }),
        prompt
      });
    } else {
      return res.status(400).json({ error: "Invalid mode" });
    }

    const output = Buffer.from(await video.arrayBuffer());

    res.setHeader("Content-Type", "video/mp4");
    res.send(output);

  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Video generation failed" });
  }
});

const PORT = process.env.PORT || 10000;
app.listen(PORT, "0.0.0.0", () => {
  console.log(`Server running on port ${PORT}`);
});
