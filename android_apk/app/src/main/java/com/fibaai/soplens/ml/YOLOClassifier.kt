package com.fibaai.soplens.ml

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.util.Collections
import kotlin.math.max
import kotlin.math.min

data class Detection(
    val classId: Int,
    val className: String,
    val confidence: Float,
    val x1: Float, // normalized to 0..1
    val y1: Float, // normalized to 0..1
    val x2: Float,
    val y2: Float
)
    
class YOLOClassifier(context: Context) {
    private val modelPath = "yolov8n.onnx"
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    // 0..79 COCO Classes mapping
    private val classes = arrayOf(
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", 
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", 
        "hair drier", "toothbrush"
    )
    
    // Map of common aliases that users might search for
    val ALIAS_MAP = mapOf(
        "mobile" to "cell phone",
        "phone" to "cell phone",
        "water" to "bottle",
        "tea" to "cup",
        "coffee" to "cup",
        "pen" to "toothbrush", // fallback mapping if YOLO misclassifies thin objects as toothbrush
        "screw" to "scissors"
    )

    init {
        val modelBytes = context.assets.open(modelPath).readBytes()
        val options = OrtSession.SessionOptions().apply {
            addConfigEntry("session.load_model_format", "ORT")
        }
        session = env.createSession(modelBytes, options)
    }

    fun detect(bitmap: Bitmap): List<Detection> {
        val width = 320
        val height = 320
        val scaled = Bitmap.createScaledBitmap(bitmap, width, height, true)
        
        val floatBuffer = FloatBuffer.allocate(1 * 3 * width * height)
        val pixels = IntArray(width * height)
        scaled.getPixels(pixels, 0, width, 0, 0, width, height)

        var idx = 0
        val rOffset = 0
        val gOffset = width * height
        val bOffset = 2 * width * height

        for (y in 0 until height) {
            for (x in 0 until width) {
                val p = pixels[y * width + x]
                floatBuffer.put(rOffset + idx, ((p shr 16 and 0xFF) / 255.0f))
                floatBuffer.put(gOffset + idx, ((p shr 8 and 0xFF) / 255.0f))
                floatBuffer.put(bOffset + idx, ((p and 0xFF) / 255.0f))
                idx++
            }
        }

        val inputName = session.inputNames.iterator().next()
        val shape = longArrayOf(1, 3, height.toLong(), width.toLong())
        val tensor = OnnxTensor.createTensor(env, floatBuffer, shape)
        
        val output = session.run(Collections.singletonMap(inputName, tensor))
        @Suppress("UNCHECKED_CAST")
        val outputArray = (output.iterator().next().value as Array<Array<FloatArray>>)[0]
        
        // Output shape for YOLOv8 is [84][2100]
        // 84 = 4 bounding box coordinates (cx, cy, w, h) + 80 class logits
        val numChannels = outputArray.size
        val numElements = outputArray[0].size
        
        val detections = mutableListOf<Detection>()
        val confThreshold = 0.25f

        for (i in 0 until numElements) {
            var maxClassConf = 0f
            var maxClassId = -1

            for (c in 0 until 80) {
                val conf = outputArray[c + 4][i]
                if (conf > maxClassConf) {
                    maxClassConf = conf
                    maxClassId = c
                }
            }

            if (maxClassConf > confThreshold) {
                val cx = outputArray[0][i]
                val cy = outputArray[1][i]
                val w = outputArray[2][i]
                val h = outputArray[3][i]

                val x1 = (cx - w / 2) / 320f
                val y1 = (cy - h / 2) / 320f
                val x2 = (cx + w / 2) / 320f
                val y2 = (cy + h / 2) / 320f

                detections.add(
                    Detection(
                        classId = maxClassId,
                        className = classes[maxClassId],
                        confidence = maxClassConf,
                        x1 = x1, y1 = y1, x2 = x2, y2 = y2
                    )
                )
            }
        }

        output.close()
        tensor.close()

        // Apply basic NMS
        return applyNMS(detections, 0.45f)
    }
    
    private fun applyNMS(detections: List<Detection>, iouThreshold: Float): List<Detection> {
        val sorted = detections.sortedByDescending { it.confidence }
        val result = mutableListOf<Detection>()
        
        for (i in sorted.indices) {
            val d1 = sorted[i]
            var keep = true
            for (j in result.indices) {
                val d2 = result[j]
                if (d1.classId == d2.classId && computeIoU(d1, d2) > iouThreshold) {
                    keep = false
                    break
                }
            }
            if (keep) {
                result.add(d1)
            }
        }
        return result
    }
    
    private fun computeIoU(d1: Detection, d2: Detection): Float {
        val ix1 = max(d1.x1, d2.x1)
        val iy1 = max(d1.y1, d2.y1)
        val ix2 = min(d1.x2, d2.x2)
        val iy2 = min(d1.y2, d2.y2)
        
        val iArea = max(0f, ix2 - ix1) * max(0f, iy2 - iy1)
        val area1 = (d1.x2 - d1.x1) * (d1.y2 - d1.y1)
        val area2 = (d2.x2 - d2.x1) * (d2.y2 - d2.y1)
        
        return iArea / (area1 + area2 - iArea)
    }

    fun close() {
        session.close()
    }
}
