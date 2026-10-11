package com.ruthlessmallard.radioscribe

import android.content.res.AssetFileDescriptor
import android.content.res.AssetManager
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.io.File
import java.io.FileOutputStream

/// Hosts a MethodChannel used by AudioService to extract large bundled model
/// files from APK assets to app storage without loading them into memory.
class MainActivity : FlutterActivity() {
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "radioscribe/model")
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "extractAsset" -> {
                        val assetPath = call.argument<String>("assetPath")
                        val destPath = call.argument<String>("destPath")
                        if (assetPath == null || destPath == null) {
                            result.error("BAD_ARGS", "assetPath and destPath are required", null)
                            return@setMethodCallHandler
                        }
                        try {
                            result.success(extractAsset(assets, assetPath, destPath))
                        } catch (e: Exception) {
                            result.error("EXTRACT_FAILED", e.message, null)
                        }
                    }
                    else -> result.notImplemented()
                }
            }
    }

    /// Streams [assetPath] out of the APK into [destPath], writing to a temp
    /// file and renaming so a crash never leaves a half-written model.
    private fun extractAsset(assets: AssetManager, assetPath: String, destPath: String): String {
        val dest = File(destPath)
        val expected = try {
            assets.openFd(assetPath).use { it.declaredLength }
        } catch (e: Exception) {
            -1L
        }
        if (expected > 0 && dest.exists() && dest.length() == expected) return destPath
        if (expected <= 0 && dest.exists() && dest.length() > 0) return destPath

        dest.parentFile?.mkdirs()
        val tmp = File("$destPath.part")
        assets.open(assetPath).use { input ->
            FileOutputStream(tmp).use { output ->
                input.copyTo(output, 1 shl 20)
            }
        }
        if (expected > 0 && tmp.length() != expected) {
            tmp.delete()
            throw IllegalStateException(
                "asset size mismatch: extracted ${tmp.length()}, expected $expected"
            )
        }
        if (dest.exists()) dest.delete()
        if (!tmp.renameTo(dest)) {
            throw IllegalStateException("rename failed: $tmp -> $dest")
        }
        return destPath
    }
}
