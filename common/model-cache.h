#pragma once

#include <cstdint>
#include <cstdio>
#include <ctime>
#include <map>
#include <string>
#include <vector>

struct model_cache_entry {
    std::string name;
    std::string path;
    uint64_t size = 0;
    std::string sha256;
    time_t added = 0;
};

std::string model_cache_join_path(const std::string & root, const std::string & rel);

class model_cache {
public:
    explicit model_cache(const std::string & root);

    bool load_manifest();
    bool save_manifest();
    std::vector<model_cache_entry> list() const;
    bool add(const std::string & name, const std::string & src_path);
    bool remove(const std::string & name);
    std::string resolve(const std::string & name) const;
    bool download(const std::string & url, const std::string & name, const std::string & auth_token);
    bool verify(const std::string & name) const;

    uint64_t total_size() const;
    size_t size() const;
    std::string root() const;
    bool contains(const std::string & name) const;
    std::vector<std::string> names() const;
    bool touch(const std::string & name);
    bool export_manifest(const std::string & path) const;
    bool import_manifest(const std::string & path);
    bool refresh();
    bool clear();
    bool has_manifest() const;
    std::string manifest_text() const;
    std::vector<model_cache_entry> find_prefix(const std::string & prefix) const;
    size_t remove_older_than(time_t cutoff);
    bool copy_to(const std::string & name, const std::string & destination) const;
    std::string path_for(const std::string & name) const;
    void set_root(const std::string & root);
    std::map<std::string, std::string> metadata(const std::string & name) const;

private:
    std::string manifest_path() const;
    std::string entry_path(const std::string & name) const;
    bool write_entry(FILE * file, const model_cache_entry & entry) const;
    bool read_entry(FILE * file, model_cache_entry & entry);
    std::string checksum(const std::string & path) const;
    bool ensure_root() const;
    static std::string trim(const std::string & value);
    static std::string basename(const std::string & value);

    std::string root_;
    std::map<std::string, model_cache_entry> entries_;
};

model_cache * model_cache_global(const std::string & root);
