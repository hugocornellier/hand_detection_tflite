#include "include/hand_detection_tflite/hand_detection_tflite_plugin_c_api.h"

#include <flutter/plugin_registrar_windows.h>

#include "hand_detection_tflite_plugin.h"

void HandDetectionTflitePluginCApiRegisterWithRegistrar(
    FlutterDesktopPluginRegistrarRef registrar) {
  hand_detection_tflite::HandDetectionTflitePlugin::RegisterWithRegistrar(
      flutter::PluginRegistrarManager::GetInstance()
          ->GetRegistrar<flutter::PluginRegistrarWindows>(registrar));
}
