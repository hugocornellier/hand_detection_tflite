#include "include/hand_detection_tflite/hand_detection_tflite_plugin.h"
#include "hand_detection_tflite_plugin.h"
#include <flutter/plugin_registrar_windows.h>

void PoseDetectionTflitePluginRegisterWithRegistrar(FlutterDesktopPluginRegistrarRef registrar) {
  auto cpp_registrar =
      flutter::PluginRegistrarManager::GetInstance()
          ->GetRegistrar<flutter::PluginRegistrarWindows>(registrar);
  hand_detection_tflite::PoseDetectionTflitePlugin::RegisterWithRegistrar(cpp_registrar);
}
