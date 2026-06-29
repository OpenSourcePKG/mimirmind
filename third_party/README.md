# Vendored third-party headers

Single-header dependencies pinned to a specific upstream tag. The build
includes this directory as a SYSTEM include path so upstream code does
not trigger the project's strict warning flags.

| Library      | Version | Source |
| ------------ | ------- | ------ |
| cpp-httplib  | v0.18.5 | https://github.com/yhirose/cpp-httplib/blob/v0.18.5/httplib.h |
| nlohmann/json| v3.11.3 | https://github.com/nlohmann/json/blob/v3.11.3/single_include/nlohmann/json.hpp |

To upgrade, fetch the matching file from the upstream tag and bump the
table above. Run the smoke test afterwards — both libraries are header-
only so an upgrade only takes effect after a rebuild.