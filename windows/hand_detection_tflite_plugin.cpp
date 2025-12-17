#include "hand_detection_tflite_plugin.h"
#include "include/hand_detection_tflite/hand_detection_tflite_plugin.h"  // ensures dllexport is seen here

#include <windows.h>
#include <VersionHelpers.h>
#include <flutter/method_channel.h>
#include <flutter/plugin_registrar_windows.h>
#include <flutter/standard_method_codec.h>

#include <memory>
#include <sstream>

namespace hand_detection_tflite {

void HandDetectionTflitePlugin::RegisterWithRegistrar(
    flutter::PluginRegistrarWindows *registrar) {
  auto channel =
      std::make_unique<flutter::MethodChannel<flutter::EncodableValue>>(
          registrar->messenger(), "hand_detection_tflite",
          &flutter::StandardMethodCodec::GetInstance());

  auto plugin = std::make_unique<HandDetectionTflitePlugin>();

  channel->SetMethodCallHandler(
      [plugin_pointer = plugin.get()](const auto &call, auto result) {
        plugin_pointer->HandleMethodCall(call, std::move(result));
      });

  registrar->AddPlugin(std::move(plugin));
}

HandDetectionTflitePlugin::HandDetectionTflitePlugin() {}
HandDetectionTflitePlugin::~HandDetectionTflitePlugin() {}

void HandDetectionTflitePlugin::HandleMethodCall(
    const flutter::MethodCall<flutter::EncodableValue> &method_call,
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {
  if (method_call.method_name().compare("getPlatformVersion") == 0) {
    std::ostringstream version_stream;
    version_stream << "Windows ";
    if (IsWindows10OrGreater()) {
      version_stream << "10+";
    } else if (IsWindows8OrGreater()) {
      version_stream << "8";
    } else if (IsWindows7OrGreater()) {
      version_stream << "7";
    }
    result->Success(flutter::EncodableValue(version_stream.str()));
  } else {
    result->NotImplemented();
  }
}

}  // namespace hand_detection_tflite

void HandDetectionTflitePluginRegisterWithRegistrar(
    FlutterDesktopPluginRegistrarRef registrar) {
  auto cpp_registrar =
      flutter::PluginRegistrarManager::GetInstance()
          ->GetRegistrar<flutter::PluginRegistrarWindows>(registrar);
  hand_detection_tflite::HandDetectionTflitePlugin::RegisterWithRegistrar(cpp_registrar);
}
