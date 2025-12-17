#ifndef FLUTTER_PLUGIN_HAND_DETECTION_TFLITE_PLUGIN_H_
#define FLUTTER_PLUGIN_HAND_DETECTION_TFLITE_PLUGIN_H_

#include <flutter/method_channel.h>
#include <flutter/plugin_registrar_windows.h>

#include <memory>

namespace hand_detection_tflite {

class HandDetectionTflitePlugin : public flutter::Plugin {
 public:
  static void RegisterWithRegistrar(flutter::PluginRegistrarWindows *registrar);

  HandDetectionTflitePlugin();

  virtual ~HandDetectionTflitePlugin();

  HandDetectionTflitePlugin(const HandDetectionTflitePlugin&) = delete;
  HandDetectionTflitePlugin& operator=(const HandDetectionTflitePlugin&) = delete;

  void HandleMethodCall(
      const flutter::MethodCall<flutter::EncodableValue> &method_call,
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);
};

}  // namespace hand_detection_tflite

#endif  // FLUTTER_PLUGIN_HAND_DETECTION_TFLITE_PLUGIN_H_
