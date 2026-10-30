import { InferenceClient } from "@huggingface/inference";

export default async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", process.env.ALLOWED_ORIGIN || "");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") return res.status(200).end();
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  try {
    const { prompt, mode, imageBase64, mimeType } = req.body || {};
    if (!process.env.HF_TOKEN) throw new Error("Missing HF_TOKEN");
    if (!prompt || typeof prompt !== "string" || prompt.length > 1500)
      return res.status(400).json({ error: "Invalid prompt" });

    const client = new InferenceClient(process.env.HF_TOKEN);
    const model = process.env.HF_VIDEO_MODEL || "Wan-AI/Wan2.2-TI2V-5B";

    let video;

    if (mode === "image") {
      if (!imageBase64 || !["image/jpeg", "image/png", "image/webp"].includes(mimeType))
        return res.status(400).json({ error: "Valid image required" });

      const image = Buffer.from(imageBase64, "base64");
      if (image.length > 3 * 1024 * 1024)
        return res.status(413).json({ error: "Image exceeds 3 MB" });

      video = await client.imageToVideo({
        model,
        inputs: new Blob([image], { type: mimeType }),
        prompt
      });
    } else if (mode === "text") {
      video = await client.textToVideo({ model, inputs: prompt });
    } else {
      return res.status(400).json({ error: "Invalid mode" });
    }

    const output = Buffer.from(await video.arrayBuffer());
    res.setHeader("Content-Type", "video/mp4");
    return res.status(200).send(output);
  } catch (error) {
    console.error(error);
    return res.status(502).json({ error: "AI generation failed. Check model/provider access and token." });
  }
}
