#include "model-cache.h"

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

const std::string cache_root = "/tmp/llama-model-cache-test";
const std::string source_path = "/tmp/llama-model-cache-test-source.gguf";

void write_source(const std::string & value) {
    std::filesystem::create_directories(cache_root);
    std::ofstream file(source_path, std::ios::binary | std::ios::trunc);
    file << value;
}

void assert_entry(const model_cache_entry & entry, const std::string & name, uint64_t size) {
    assert(entry.name == name);
    assert(entry.size == size);
    assert(!entry.path.empty());
    assert(!entry.sha256.empty());
    assert(entry.added > 0);
}

void test_empty_cache() {
    model_cache cache(cache_root + "/empty");
    assert(cache.list().empty());
    assert(cache.size() == 0);
    assert(cache.total_size() == 0);
    assert(!cache.contains("missing"));
    assert(cache.resolve("missing").find("missing") != std::string::npos);
}

void test_add_and_list() {
    write_source("small model payload");
    model_cache cache(cache_root + "/add");
    assert(cache.add("small.gguf", source_path));
    const auto entries = cache.list();
    assert(entries.size() == 1);
    assert_entry(entries.front(), "small.gguf", 19);
    assert(cache.contains("small.gguf"));
    assert(cache.size() == 1);
    assert(cache.total_size() == 19);
    assert(std::filesystem::exists(cache.resolve("small.gguf")));
}

void test_manifest_round_trip() {
    write_source("manifest payload");
    model_cache cache(cache_root + "/manifest");
    assert(cache.add("manifest.gguf", source_path));
    assert(cache.save_manifest());
    model_cache reloaded(cache_root + "/manifest");
    assert(reloaded.load_manifest());
    assert(reloaded.contains("manifest.gguf"));
    assert(reloaded.verify("manifest.gguf"));
    assert(reloaded.names().size() == 1);
}

void test_resolve() {
    model_cache cache(cache_root + "/resolve");
    const auto outside = cache.resolve("../../etc/passwd");
    assert(outside.find("../../etc/passwd") != std::string::npos);
    assert(model_cache_join_path("/var/models", "example.gguf") == "/var/models/example.gguf");
    assert(model_cache_join_path("/var/models/", "example.gguf") == "/var/models/example.gguf");
    assert(cache.resolve("nested/file.gguf").find("nested/file.gguf") != std::string::npos);
}

void test_remove() {
    write_source("remove payload");
    model_cache cache(cache_root + "/remove");
    assert(cache.add("remove.gguf", source_path));
    const auto path = cache.resolve("remove.gguf");
    assert(std::filesystem::exists(path));
    assert(cache.remove("remove.gguf"));
    assert(!cache.contains("remove.gguf"));
    assert(!std::filesystem::exists(path));
    assert(!cache.remove("remove.gguf"));
}

void test_export_import() {
    write_source("export payload");
    model_cache cache(cache_root + "/export");
    assert(cache.add("export.gguf", source_path));
    const auto manifest = cache_root + "/external-manifest.txt";
    assert(cache.export_manifest(manifest));
    model_cache imported(cache_root + "/imported");
    assert(imported.import_manifest(manifest));
    assert(imported.contains("export.gguf"));
    assert(imported.list().front().name == "export.gguf");
}

void test_touch() {
    write_source("touch payload");
    model_cache cache(cache_root + "/touch");
    assert(cache.add("touch.gguf", source_path));
    const auto before = cache.list().front().added;
    assert(cache.touch("touch.gguf"));
    const auto after = cache.list().front().added;
    assert(after >= before);
    assert(!cache.touch("missing"));
}

void test_paths() {
    model_cache cache(cache_root + "/paths");
    const auto first = cache.resolve("one.gguf");
    const auto second = cache.resolve("/absolute/path.gguf");
    assert(first.front() == '/');
    assert(second == cache_root + "/paths//absolute/path.gguf");
    assert(cache.root() == cache_root + "/paths");
}

void test_multiple_entries() {
    write_source("multiple payload");
    model_cache cache(cache_root + "/multiple");
    for (int i = 0; i < 4; ++i) {
        assert(cache.add("model-" + std::to_string(i) + ".gguf", source_path));
    }
    assert(cache.list().size() == 4);
    assert(cache.names().size() == 4);
    assert(cache.total_size() == 4 * 16);
}

void test_missing_files() {
    model_cache cache(cache_root + "/missing");
    assert(!cache.add("none.gguf", cache_root + "/not-here.gguf"));
    assert(!cache.verify("none.gguf"));
    assert(!cache.remove("none.gguf"));
}

void test_search_and_metadata() {
    write_source("search payload");
    model_cache cache(cache_root + "/search");
    assert(cache.add("alpha.gguf", source_path));
    assert(cache.add("alphabet.gguf", source_path));
    assert(cache.add("beta.gguf", source_path));
    const auto matches = cache.find_prefix("alph");
    assert(matches.size() == 2);
    const auto metadata = cache.metadata("alpha.gguf");
    assert(metadata.at("name") == "alpha.gguf");
    assert(metadata.at("path") == "alpha.gguf");
    assert(metadata.at("size") == "14");
    assert(metadata.count("sha256") == 1);
}

void test_copy_and_refresh() {
    write_source("copy payload");
    model_cache cache(cache_root + "/copy");
    assert(cache.add("copy.gguf", source_path));
    const auto destination = cache_root + "/copy-result.gguf";
    assert(cache.copy_to("copy.gguf", destination));
    assert(std::filesystem::exists(destination));
    assert(cache.refresh());
    assert(cache.has_manifest());
    assert(cache.contains("copy.gguf"));
}

void test_clear() {
    write_source("clear payload");
    model_cache cache(cache_root + "/clear");
    assert(cache.add("clear-a.gguf", source_path));
    assert(cache.add("clear-b.gguf", source_path));
    assert(cache.clear());
    assert(cache.list().empty());
    assert(cache.total_size() == 0);
}

} // namespace

int main() {
    test_empty_cache();
    test_add_and_list();
    test_manifest_round_trip();
    test_resolve();
    test_remove();
    test_export_import();
    test_touch();
    test_paths();
    test_multiple_entries();
    test_missing_files();
    test_search_and_metadata();
    test_copy_and_refresh();
    test_clear();
    std::printf("model cache tests passed\n");
    return 0;
}
