package com.example.mlmodeltestfinal

import android.os.*
import androidx.appcompat.app.AppCompatActivity
import com.example.mlmodeltestfinal.databinding.ActivityMainBinding
import com.example.mlmodeltestfinal.ml.GlacialLakeRiskModelupdated
import kotlinx.coroutines.*
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONArray
import org.json.JSONObject
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val client = OkHttpClient()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize binding
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.fetchButton.setOnClickListener {
            val lakeName = binding.editText.text.toString().trim()
            if (lakeName.isEmpty()) {
                binding.textView2.text = "❗ Please enter a lake name"
                return@setOnClickListener
            }

            CoroutineScope(Dispatchers.IO).launch {
                try {
                    runOnUiThread {
                        binding.textView2.text = "⏳ Fetching data for \"$lakeName\"...\n"
                        binding.progressBar.progress = 0
                        binding.progressText.text = "Progress: 0%"
                    }

                    val json = fetchJsonFromServer(lakeName)
                    val jsonArray = JSONArray(json)

                    if (jsonArray.length() == 0) {
                        runOnUiThread {
                            binding.textView2.text = "❌ No data found for $lakeName."
                        }
                        return@launch
                    }

                    val firstEntry = jsonArray.getJSONObject(0)
                    val input = extractLakeInput(firstEntry)
                    val result = runModel(input)

                    runOnUiThread {
                        binding.textView2.append("\n$result")
                        binding.progressBar.progress = 1
                        binding.progressText.text = "Progress: 100%"
                    }

                } catch (e: Exception) {
                    runOnUiThread {
                        binding.textView2.text = "❌ Error: ${e.message}"
                    }
                }
            }
        }
    }

    private fun fetchJsonFromServer(lakeName: String): String {
        val request = Request.Builder()
            .url("http://100.100.11.186:3000/$lakeName")
            .build()

        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) throw Exception("HTTP ${response.code}")
            return response.body?.string() ?: throw Exception("Empty response")
        }
    }

    // Updated this function to match NEW parameters
    private fun extractLakeInput(obj: JSONObject): FloatArray {
        val importantFeatures = floatArrayOf(
            obj.getDouble("Rainfall_Intensity_mmhr").toFloat(),
            obj.getDouble("Air_Temp_Change_Cday").toFloat(),
            obj.getDouble("Snowmelt_Rate_mmday").toFloat(),
            obj.getDouble("Lake_Water_Level_Change_mday").toFloat(),
            obj.getDouble("Seismic_Activity_Mag").toFloat(),
            obj.getDouble("Ice_Cracking_Signal").toFloat(),
            obj.getDouble("Glacier_Calving_Event").toFloat(),
            obj.getDouble("Landslide_Event").toFloat(),
            obj.getDouble("River_Discharge_Change_pct").toFloat(),
            obj.getDouble("Water_Turbidity_Index").toFloat()
        )

        // Prepare input buffer for model (size = 9813)
        val inputArray = FloatArray(9813) { 0f }

        for (i in importantFeatures.indices) {
            inputArray[i] = importantFeatures[i]
        }

        return inputArray
    }

    private fun runModel(input: FloatArray): String {
        val model = GlacialLakeRiskModelupdated.newInstance(this)

        val byteBuffer = ByteBuffer.allocateDirect(4 * 9813)
        byteBuffer.order(ByteOrder.nativeOrder())

        for (i in 0 until 9813) {
            byteBuffer.putFloat(input[i])
        }

        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 9813), DataType.FLOAT32)
        inputFeature0.loadBuffer(byteBuffer)

        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer

        model.close()

        val predictions = outputFeature0.floatArray
        val maxIndex = predictions.indices.maxByOrNull { predictions[it] } ?: -1

        return when (maxIndex) {
            0 -> "✅ Low Risk (Confidence: ${"%.2f".format(predictions[0] * 100)}%)"
            1 -> "⚠️ Moderate Risk (Confidence: ${"%.2f".format(predictions[1] * 100)}%)"
            2 -> "🚨 High Risk (Confidence: ${"%.2f".format(predictions[2] * 100)}%)"
            else -> "❓ Unknown Risk"
        }
    }
}
