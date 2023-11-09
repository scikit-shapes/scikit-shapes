"""Generate the code reference pages and navigation."""

from pathlib import Path
import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

for path in sorted(Path("skshapes").rglob("*.py")):
    module_path = path.relative_to("skshapes").with_suffix("")
    doc_path = path.relative_to("skshapes").with_suffix(".md")

    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
    elif parts[-1] == "__main__":
        continue

    # parts is the tuple of the path to the module from the skshapes folder
    if len(parts) > 0:
        nav[parts] = doc_path.as_posix()  #

        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            ident = "skshapes." + ".".join(parts)
            fd.write(f"::: {ident}")

        mkdocs_gen_files.set_edit_path(full_doc_path, Path("../") / path)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:  #
    nav_file.writelines(nav.build_literate_nav())  #
