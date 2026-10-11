import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_whisper_ggml/flutter_whisper.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:record/record.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

import 'mining_vocabulary.dart';

/// AudioService — offline STT with whisper.cpp (large-v3-turbo, fp16).
///
/// Architecture (whisper is not streaming, so this is a windowed pipeline):
///   1. On first [startListening], the bundled GGML model is streamed out of
///      the APK into app storage (one-time, via a platform channel so the
///      1.5 GB file never enters Dart memory), then loaded into whisper.cpp.
///   2. Mic audio (PCM 16-bit, 16 kHz, mono) is bandpass-filtered
///      (300-3400 Hz) and run through an energy VAD.
///   3. Speech is buffered until ~1.2 s of silence or a 30 s cap, then the
///      window is transcribed. Transcriptions are serialized by the plugin.
///   4. Finished transcripts are emitted on [segmentStream].
///   5. [stopListening] flushes any buffered speech and stops the recorder.
///
/// Fully offline: the model ships inside the APK; no network is used at any
/// point (the manifest deliberately has no INTERNET permission).
class AudioService extends ChangeNotifier {
  // ── Public state ──────────────────────────────────────────────────────────
  bool _isListening = false;
  bool _hasPermission = false;
  String _partialText = '';
  bool _modelLoaded = false;
  bool _modelError = false;
  String _modelErrorMessage = '';

  bool get isListening => _isListening;
  bool get hasPermission => _hasPermission;
  String get partialText => _partialText;
  bool get modelLoaded => _modelLoaded;
  bool get modelError => _modelError;
  String get modelErrorMessage => _modelErrorMessage;

  /// Emits completed transcript segments.
  Stream<String> get segmentStream => _segmentController.stream;

  // ── Configuration ─────────────────────────────────────────────────────────
  static const int _sampleRate = 16000;

  /// Bundled model asset inside the APK (android/app/src/main/assets).
  static const String _modelAssetPath = 'models/ggml-large-v3-turbo.bin';
  static const String _modelFileName = 'ggml-large-v3-turbo.bin';

  /// Platform channel used to stream the model asset to disk without
  /// loading it into Dart memory. Implemented in MainActivity.
  static const MethodChannel _modelChannel = MethodChannel('radioscribe/model');

  /// Seconds of silence that ends a segment.
  static const double _silenceCutSeconds = 1.2;

  /// Maximum segment length (whisper works on ~30 s windows).
  static const double _maxSegmentSeconds = 30.0;

  /// Minimum buffered speech before a segment is worth transcribing.
  static const double _minSegmentSeconds = 0.5;

  /// Default speech-detection threshold (RMS) when the user's energy gate
  /// setting is 0 (disabled).
  static const double _defaultSpeechThreshold = 0.004;

  /// Speech-detection threshold. The existing energy-gate setting doubles as
  /// the VAD threshold; 0 = use the field default.
  double _energyGateThreshold = 0.0;

  set energyGateThreshold(double value) {
    _energyGateThreshold = value;
  }

  double get energyGateThreshold => _energyGateThreshold;

  double get _speechThreshold =>
      _energyGateThreshold > 0 ? _energyGateThreshold : _defaultSpeechThreshold;

  /// Transcription configuration: English radio traffic, biased toward the
  /// mining vocabulary, with hallucination guards tuned for noise.
  final WhisperTranscribeConfig _transcribeConfig = WhisperTranscribeConfig(
    language: 'en',
    threads: 4,
    noContext: true, // don't let one bad window poison the next
    initialPrompt: MiningVocabulary.whisperInitialPrompt,
    temperature: 0.0,
    noSpeechThreshold: 0.6,
    logProbabilityThreshold: -1.0,
  );

  // ── Internals ─────────────────────────────────────────────────────────────
  final StreamController<String> _segmentController =
      StreamController<String>.broadcast();

  FlutterWhisper? _whisper;

  // Record package
  AudioRecorder? _recorder;
  StreamSubscription<Uint8List>? _audioSub;

  // VAD / segmentation state
  final List<Float32List> _bufferChunks = <Float32List>[];
  int _bufferSamples = 0;
  bool _hasSpeech = false;
  double _silenceSeconds = 0.0;
  int _pendingTranscriptions = 0;

  // Bandpass filter state (biquad cascade: high-pass @ 300Hz, low-pass @ 3400Hz)
  double _hpX1 = 0, _hpX2 = 0, _hpY1 = 0, _hpY2 = 0;
  double _lpX1 = 0, _lpX2 = 0, _lpY1 = 0, _lpY2 = 0;

  // ── Permission ────────────────────────────────────────────────────────────

  Future<bool> requestPermission() async {
    final status = await Permission.microphone.request();
    _hasPermission = status.isGranted;
    notifyListeners();
    return _hasPermission;
  }

  // ── Model loading ─────────────────────────────────────────────────────────

  /// Streams the bundled GGML model out of the APK into app storage and
  /// returns its path. Cached: identical size is reused on later launches.
  Future<String> _extractModelAsset() async {
    final support = await getApplicationSupportDirectory();
    final dest = p.join(support.path, _modelFileName);
    final result = await _modelChannel.invokeMethod<String>('extractAsset', {
      'assetPath': _modelAssetPath,
      'destPath': dest,
    });
    final path = result ?? dest;
    final file = File(path);
    if (!await file.exists() || await file.length() < 1024 * 1024) {
      throw Exception('Model extraction failed ($path)');
    }
    return path;
  }

  Future<void> _ensureModelLoaded() async {
    if (_modelLoaded || _modelError) return;
    try {
      debugPrint('[AudioService] Extracting whisper model from APK...');
      final modelPath = await _extractModelAsset();

      debugPrint('[AudioService] Loading model into whisper.cpp...');
      _whisper = await FlutterWhisper.load(
        modelPath: modelPath,
        config: const WhisperModelConfig(useGpu: false),
        onModelProgress: (progress) =>
            debugPrint('[AudioService] model load $progress%'),
      );

      _modelLoaded = true;
      debugPrint('[AudioService] whisper.cpp model loaded.');
    } catch (e, st) {
      debugPrint('[AudioService] Model load failed: $e\n$st');
      _modelError = true;
      _modelErrorMessage = e.toString();
    }
    notifyListeners();
  }

  // ── Listening lifecycle ───────────────────────────────────────────────────

  Future<void> startListening() async {
    if (_isListening) return;

    if (!_hasPermission) {
      final granted = await requestPermission();
      if (!granted) return;
    }

    await _ensureModelLoaded();
    if (_modelError || _whisper == null) {
      debugPrint('[AudioService] Cannot start: model failed to load.');
      return;
    }

    _recorder = AudioRecorder();

    try {
      final audioStream = await _recorder!.startStream(
        const RecordConfig(
          encoder: AudioEncoder.pcm16bits,
          sampleRate: _sampleRate,
          numChannels: 1,
        ),
      );

      _isListening = true;
      notifyListeners();

      // Prevent screen sleep while monitoring
      try {
        await WakelockPlus.enable();
      } catch (_) {}

      _audioSub = audioStream.listen(
        _onAudioChunk,
        onError: (Object e) {
          debugPrint('[AudioService] Audio stream error: $e');
        },
        cancelOnError: false,
      );
    } catch (e) {
      debugPrint('[AudioService] Failed to start recorder: $e');
      _recorder = null;
    }
  }

  Future<void> stopListening() async {
    if (!_isListening) return;

    _isListening = false;

    await _audioSub?.cancel();
    _audioSub = null;

    try {
      await _recorder?.stop();
    } catch (_) {}
    _recorder = null;

    // Transcribe whatever speech is still buffered.
    _flushSegment();

    notifyListeners();

    try {
      await WakelockPlus.disable();
    } catch (_) {}
  }

  // ── Audio processing ──────────────────────────────────────────────────────

  void _onAudioChunk(Uint8List data) {
    try {
      var samplesFloat32 = _convertBytesToFloat32(data);
      // Apply 300-3400Hz bandpass filter to match radio frequency range.
      samplesFloat32 = _applyBandpassFilter(samplesFloat32);

      final duration = samplesFloat32.length / _sampleRate;
      final isSpeech = _isEnergyAboveThreshold(
        samplesFloat32,
        _speechThreshold,
      );

      // Buffer from the moment speech starts (including trailing silence so
      // the model hears natural pauses), until the cut or cap is reached.
      if (isSpeech || _hasSpeech) {
        _bufferChunks.add(samplesFloat32);
        _bufferSamples += samplesFloat32.length;

        if (isSpeech) {
          _hasSpeech = true;
          _silenceSeconds = 0;
        } else if (_hasSpeech) {
          _silenceSeconds += duration;
        }

        final bufferedSeconds = _bufferSamples / _sampleRate;
        if (_hasSpeech &&
            (_silenceSeconds >= _silenceCutSeconds ||
                bufferedSeconds >= _maxSegmentSeconds)) {
          _flushSegment();
          notifyListeners();
        }
      }
    } catch (e) {
      debugPrint('[AudioService] Audio processing error: $e');
    }
  }

  // ── Segmentation ──────────────────────────────────────────────────────────

  /// Concatenates the buffered window and queues it for transcription.
  void _flushSegment() {
    final seconds = _bufferSamples / _sampleRate;
    if (!_hasSpeech || seconds < _minSegmentSeconds) {
      _clearBuffer();
      return;
    }

    final samples = Float32List(_bufferSamples);
    var offset = 0;
    for (final chunk in _bufferChunks) {
      samples.setRange(offset, offset + chunk.length, chunk);
      offset += chunk.length;
    }
    _clearBuffer();

    unawaited(_transcribeSegment(samples));
  }

  void _clearBuffer() {
    _bufferChunks.clear();
    _bufferSamples = 0;
    _hasSpeech = false;
    _silenceSeconds = 0;
  }

  Future<void> _transcribeSegment(Float32List samples) async {
    final whisper = _whisper;
    if (whisper == null) return;

    _pendingTranscriptions++;
    if (_partialText.isEmpty) {
      _partialText = '…'; // in-flight indicator
      notifyListeners();
    }

    try {
      // The plugin serializes concurrent transcriptions internally.
      final result = await whisper.transcribe(
        samples,
        config: _transcribeConfig,
      );
      final text = result.text.trim();
      if (text.isNotEmpty) {
        _segmentController.add(text);
      }
    } catch (e) {
      debugPrint('[AudioService] Transcription failed: $e');
    } finally {
      _pendingTranscriptions--;
      if (_pendingTranscriptions == 0 && _partialText.isNotEmpty) {
        _partialText = '';
        notifyListeners();
      }
    }
  }

  // ── PCM helpers ───────────────────────────────────────────────────────────

  Float32List _convertBytesToFloat32(Uint8List bytes,
      [Endian endian = Endian.little]) {
    // Use offset-aware ByteData view — stream chunks may have non-zero offsetInBytes
    final values = Float32List(bytes.lengthInBytes ~/ 2);
    final data = ByteData.view(
        bytes.buffer, bytes.offsetInBytes, bytes.lengthInBytes);
    for (var i = 0; i < bytes.lengthInBytes; i += 2) {
      final short = data.getInt16(i, endian);
      values[i ~/ 2] = short / 32768.0;
    }
    return values;
  }

  // ── Bandpass filter (300-3400Hz) ───────────────────────────────────────────

  /// Apply 2nd-order Butterworth bandpass filter (cascade of HP @ 300Hz + LP @ 3400Hz).
  /// Tuned for 16kHz sample rate to match typical radio voice range.
  Float32List _applyBandpassFilter(Float32List samples) {
    // Coefficients pre-calculated for Butterworth @ 16kHz
    // High-pass @ 300Hz (2nd order)
    const double hpB0 = 0.9706196;
    const double hpB1 = -1.9412393;
    const double hpB2 = 0.9706196;
    const double hpA1 = -1.9409320;
    const double hpA2 = 0.9415465;

    // Low-pass @ 3400Hz (2nd order)
    const double lpB0 = 0.2075319;
    const double lpB1 = 0.4150638;
    const double lpB2 = 0.2075319;
    const double lpA1 = -0.3139590;
    const double lpA2 = 0.1440860;

    final output = Float32List(samples.length);

    for (var i = 0; i < samples.length; i++) {
      final x = samples[i];

      // High-pass stage
      final hpOut = hpB0 * x + hpB1 * _hpX1 + hpB2 * _hpX2 - hpA1 * _hpY1 - hpA2 * _hpY2;
      _hpX2 = _hpX1;
      _hpX1 = x;
      _hpY2 = _hpY1;
      _hpY1 = hpOut;

      // Low-pass stage
      final lpOut = lpB0 * hpOut + lpB1 * _lpX1 + lpB2 * _lpX2 - lpA1 * _lpY1 - lpA2 * _lpY2;
      _lpX2 = _lpX1;
      _lpX1 = hpOut;
      _lpY2 = _lpY1;
      _lpY1 = lpOut;

      output[i] = lpOut;
    }

    return output;
  }

  /// Check if frame energy (RMS) is above the speech threshold.
  bool _isEnergyAboveThreshold(Float32List samples, double threshold) {
    if (samples.isEmpty) return false;
    double sumSquares = 0.0;
    for (final s in samples) {
      sumSquares += s * s;
    }
    final rms = math.sqrt(sumSquares / samples.length);
    return rms >= threshold;
  }

  // ── Dispose ───────────────────────────────────────────────────────────────

  @override
  void dispose() {
    _audioSub?.cancel();
    _recorder?.stop().catchError((_) => null);
    unawaited(_whisper?.dispose() ?? Future.value());
    _segmentController.close();
    super.dispose();
  }
}
