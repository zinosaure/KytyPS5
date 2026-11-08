#include "common/additionalContent.h"

#include "common/stringUtils.h"

#include <algorithm>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <regex>
#include <system_error>

namespace Common {

namespace {

std::string GetString(const nlohmann::json& object, const std::string& key) {
	const auto it = object.find(key);
	return it != object.end() && it->is_string() ? it->get<std::string>() : std::string {};
}

std::string GetTitle(const nlohmann::json& param) {
	const auto localized = param.find("localizedParameters");
	if (localized != param.end() && localized->is_object()) {
		const auto preferred = localized->find(GetString(*localized, "defaultLanguage"));
		if (preferred != localized->end()) {
			if (auto title = GetString(*preferred, "titleName"); !title.empty()) {
				return title;
			}
		}
		for (const auto& language: *localized) {
			if (auto title = GetString(language, "titleName"); !title.empty()) {
				return title;
			}
		}
	}
	return GetString(param, "titleName");
}

} // namespace

std::vector<AddContentEntry> FindAddContent(const std::filesystem::path& dir,
                                            const std::string&           title_id) {
	if (dir.empty() || title_id.empty()) {
		return {};
	}

	std::error_code ec;
	const auto      root = std::filesystem::absolute(dir, ec);
	if (ec) {
		return {};
	}

	// Scan directories separately so an unreadable child does not discard its siblings.
	std::vector<std::filesystem::path> pending {root};
	std::vector<std::filesystem::path> params;
	while (!pending.empty()) {
		const auto current = std::move(pending.back());
		pending.pop_back();
		std::filesystem::directory_iterator it(
		    current, std::filesystem::directory_options::skip_permission_denied, ec);
		const std::filesystem::directory_iterator end;
		for (; !ec && it != end; it.increment(ec)) {
			std::error_code entry_ec;
			if (it->is_directory(entry_ec) && !it->is_symlink(entry_ec) && !entry_ec) {
				pending.push_back(it->path());
			} else if (it->is_regular_file(entry_ec) && !entry_ec &&
			           EqualNoCase(PathToString(it->path().filename()), "param.json") &&
			           EqualNoCase(PathToString(it->path().parent_path().filename()), "sce_sys")) {
				params.push_back(it->path());
			}
		}
	}
	std::sort(params.begin(), params.end());

	static const std::regex content_id_pattern(
	    "^[A-Z]{2}[0-9]{4}-[A-Z]{4}[0-9]{5}_[0-9]{2}-[A-Z0-9]{16}$");
	std::map<std::string, AddContentEntry> by_label;
	for (const auto& file: params) {
		const auto size = std::filesystem::file_size(file, ec);
		if (ec || size > 1024 * 1024) {
			continue;
		}
		std::ifstream stream(file, std::ios::binary);
		if (!stream) {
			continue;
		}
		const auto param = nlohmann::json::parse(stream, nullptr, false);
		// applicationCategoryType is required for applications, not additional content.
		if (!param.is_object() || param.contains("applicationCategoryType")) {
			continue;
		}
		const auto content_id = GetString(param, "contentId");
		if (!std::regex_match(content_id, content_id_pattern) ||
		    content_id.compare(7, 9, title_id) != 0) {
			continue;
		}

		const auto mount_dir = file.parent_path().parent_path();
		const auto label     = content_id.substr(20);
		by_label.try_emplace(label,
		                     AddContentEntry {content_id, label, mount_dir, GetTitle(param)});
	}

	std::vector<AddContentEntry> entries;
	entries.reserve(by_label.size());
	for (auto& [label, entry]: by_label) {
		entries.push_back(std::move(entry));
	}
	return entries;
}

std::vector<AddContentEntry> FindDeclaredAddContent(const std::filesystem::path& app0_dir) {
	std::vector<AddContentEntry> entries;
	std::ifstream                stream(app0_dir / "dlc_emu.ini", std::ios::binary);
	if (!stream) {
		return entries;
	}

	const auto trim = [](std::string_view text) {
		const auto first = text.find_first_not_of(" \t\r");
		const auto last  = text.find_last_not_of(" \t\r");
		return first == std::string_view::npos ? std::string_view {}
		                                       : text.substr(first, last - first + 1);
	};
	struct Section {
		std::string content_id;
		std::string status;
		std::string mount_point;
	};
	std::vector<Section> sections;
	std::string          line;
	while (std::getline(stream, line)) {
		const auto text = trim(line);
		if (text.starts_with('[')) {
			sections.emplace_back();
			continue;
		}
		const auto eq = text.find('=');
		if (eq == std::string_view::npos || sections.empty()) {
			continue;
		}
		const auto key   = trim(text.substr(0, eq));
		const auto value = std::string(trim(text.substr(eq + 1)));
		if (key == "content_id") {
			sections.back().content_id = value;
		} else if (key == "download_status") {
			sections.back().status = value;
		} else if (key == "mount_point") {
			sections.back().mount_point = value;
		}
	}

	static const std::regex     label_pattern("^[A-Z0-9]{16}$");
	constexpr std::string_view app0 = "/app0/";
	for (const auto& section: sections) {
		const auto dash  = section.content_id.rfind('-');
		const auto label = dash == std::string::npos ? std::string {} : section.content_id.substr(dash + 1);
		if (!std::regex_match(label, label_pattern)) {
			continue;
		}
		AddContentEntry entry {section.content_id, label, {}, {}};
		const std::string_view mount = section.mount_point;
		if (EqualNoCase(section.status, "INSTALLED") && mount.starts_with(app0) &&
		    mount.find("..") == std::string_view::npos) {
			auto            dir = app0_dir / PathFromUtf8(mount.substr(app0.size()));
			std::error_code ec;
			if (std::filesystem::is_directory(dir, ec)) {
				entry.mount_dir = std::move(dir);
			}
		}
		entries.push_back(std::move(entry));
	}
	return entries;
}

std::optional<AddContentEntry> ParseAddContent(std::string_view value) {
	static const std::regex label_pattern("^[A-Z0-9]{16}$");
	const auto              eq    = value.find('=');
	const auto              label = value.substr(0, eq);
	if (!std::regex_match(label.begin(), label.end(), label_pattern)) {
		return std::nullopt;
	}

	AddContentEntry entry {{}, std::string(label), {}, {}};
	if (eq != std::string_view::npos) {
		entry.mount_dir = PathFromUtf8(FixFilenameSlash(value.substr(eq + 1)));
		std::error_code ec;
		if (!std::filesystem::is_directory(entry.mount_dir, ec)) {
			return std::nullopt;
		}
	}
	return entry;
}

} // namespace Common
