// swift-tools-version: 6.0
// relay's control plane in Swift, with the C++ engine built in for single-process use.
// The engine itself is built by CMake for workers (OpenMP, CUDA); here it is compiled by
// SwiftPM's clang without OpenMP, so the in-process CPU backend runs single-threaded.
import PackageDescription

let package = Package(
  name: "relay",
  platforms: [.macOS(.v15)],
  products: [
    .executable(name: "relay-server", targets: ["relay-server"]),
    .library(name: "RelayControl", targets: ["RelayControl"]),
  ],
  targets: [
    .target(
      name: "RelayEngine",
      path: "engine",
      exclude: ["src/cuda_backend.cu"],
      sources: ["src"],
      publicHeadersPath: "include",
      cxxSettings: [.unsafeFlags(["-O2", "-Wno-unknown-pragmas"])]
    ),
    .target(
      name: "RelayBridge",
      dependencies: ["RelayEngine"],
      path: "bridge",
      sources: ["src"],
      publicHeadersPath: "include",
      cxxSettings: [.unsafeFlags(["-O2"])]
    ),
    .target(name: "RelayTokenizer", path: "control/Sources/RelayTokenizer"),
    .target(
      name: "RelayControl",
      dependencies: ["RelayTokenizer", "RelayBridge"],
      path: "control/Sources/RelayControl",
      swiftSettings: [.interoperabilityMode(.Cxx)]
    ),
    .executableTarget(
      name: "relay-server",
      dependencies: ["RelayControl"],
      path: "control/Sources/relay-server",
      swiftSettings: [.interoperabilityMode(.Cxx)]
    ),
    .testTarget(
      name: "RelayTokenizerTests",
      dependencies: ["RelayTokenizer"],
      path: "control/Tests/RelayTokenizerTests"
    ),
    .testTarget(
      name: "RelayControlTests",
      dependencies: ["RelayControl"],
      path: "control/Tests/RelayControlTests",
      swiftSettings: [.interoperabilityMode(.Cxx)]
    ),
  ],
  cxxLanguageStandard: .cxx20
)
