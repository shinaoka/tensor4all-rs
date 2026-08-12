#!/usr/bin/env python3
"""
Refactor treetn modules from TreeTN<I, V> to TreeTN<T, V> pattern.

Pattern changes:
- impl<I, V> TreeTN<I, V> where I: IndexLike -> impl<T, V> TreeTN<T, V> where T: TensorLike
- IdxTensor<I::Id, I::Symm> -> T
- I (as index type in TreeTN context) -> T::Index
- I::Id -> <T::Index as IndexLike>::Id
- I::Symm -> (remove entirely)
- tensor.indices -> tensor.external_indices()
- index.id (field) -> index.id() (method)
- Remove Symmetry trait bounds

NOTE: Helper structs like MergedBondInfo<I> that only hold indices should keep I: IndexLike
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple, Set


class RustRefactorer:
    def __init__(self):
        self.changes_made = []
        # Track helper struct names that should keep I: IndexLike
        self.helper_structs: Set[str] = set()

    def refactor_file(self, filepath: Path) -> str:
        """Refactor a single Rust file."""
        content = filepath.read_text()
        original = content

        # Track what we changed
        self.changes_made = []
        self.helper_structs = set()

        # First pass: identify helper structs that should keep I: IndexLike
        self.identify_helper_structs(content)

        # Apply transformations in order
        content = self.fix_impl_blocks(content)
        content = self.fix_type_params(content)
        content = self.fix_tensor_dynlen(content)
        content = self.fix_index_type_in_treetn_context(content)
        content = self.fix_id_access(content)
        content = self.fix_tensor_indices_access(content)
        content = self.fix_index_id_field_access(content)
        content = self.remove_symm_and_tags_bounds(content)
        content = self.fix_imports(content)
        content = self.fix_where_clauses(content)
        content = self.fix_remaining_i_references(content)

        return content

    def identify_helper_structs(self, content: str) -> None:
        """Identify struct definitions with <I> where I: IndexLike."""
        # Match: pub struct SomeName<I>\n where\n  I: IndexLike
        pattern = r'pub struct (\w+)<I>\s*\n\s*where\s*\n?\s*I:\s*IndexLike'
        matches = re.findall(pattern, content)
        self.helper_structs.update(matches)

        # Also match single-line where clauses
        pattern = r'pub struct (\w+)<I>\s+where\s+I:\s*IndexLike'
        matches = re.findall(pattern, content)
        self.helper_structs.update(matches)

    def fix_impl_blocks(self, content: str) -> str:
        """Fix impl<I, V> TreeTN<I, V> patterns."""
        # impl<I, V> TreeTN<I, V> -> impl<T, V> TreeTN<T, V>
        pattern = r'impl<I,\s*V>\s+TreeTN<I,\s*V>'
        replacement = r'impl<T, V> TreeTN<T, V>'
        content = re.sub(pattern, replacement, content)

        # impl<I, V> SomeStruct<I, V> for TreeTN<I, V>
        pattern = r'impl<I,\s*V>\s+(\w+)<I,\s*V>\s+for\s+TreeTN<I,\s*V>'
        replacement = r'impl<T, V> \\1<T, V> for TreeTN<T, V>'
        content = re.sub(pattern, replacement, content)

        return content

    def fix_type_params(self, content: str) -> str:
        """Fix type parameters in where clauses for TreeTN impl blocks only."""
        # Check if this file has TreeTN impls
        if 'impl<T, V> TreeTN<T, V>' in content:
            # Replace I: IndexLike with T: TensorLike in where clauses
            pattern = r'(where\s+)I:\s*IndexLike\s*,'
            replacement = r'\1T: TensorLike,'
            content = re.sub(pattern, replacement, content)

            # I: IndexLike at end of where clause
            pattern = r'(where[^{]*?)I:\s*IndexLike\s*(\n\s*\{)'
            replacement = r'\1T: TensorLike\2'
            content = re.sub(pattern, replacement, content)

        return content

    def fix_tensor_dynlen(self, content: str) -> str:
        """Replace IdxTensor<I::Id, I::Symm> with T."""
        # IdxTensor<I::Id, I::Symm> -> T
        pattern = r'IdxTensor<I::Id,\s*I::Symm>'
        replacement = r'T'
        content = re.sub(pattern, replacement, content)

        # Vec<IdxTensor<I::Id, I::Symm>> -> Vec<T>
        pattern = r'Vec<IdxTensor<I::Id,\s*I::Symm>>'
        replacement = r'Vec<T>'
        content = re.sub(pattern, replacement, content)

        # HashMap<..., IdxTensor<I::Id, I::Symm>> -> HashMap<..., T>
        pattern = r'HashMap<([^,]+),\s*IdxTensor<I::Id,\s*I::Symm>>'
        replacement = r'HashMap<\1, T>'
        content = re.sub(pattern, replacement, content)

        return content

    def fix_index_type_in_treetn_context(self, content: str) -> str:
        """Replace I with T::Index only in TreeTN impl method contexts."""
        # Only do these replacements if we're in a TreeTN context
        if 'impl<T, V> TreeTN<T, V>' not in content:
            return content

        # DON'T replace helper struct type parameters like MergedBondInfo<I>
        # DO replace return types and usage like -> Result<HashMap<..., MergedBondInfo<I>>>
        for struct_name in self.helper_structs:
            # In return types and expressions (not struct definitions)
            pattern = rf'([>)\]]\s*{struct_name}<)I(>)'
            replacement = rf'\1T::Index\2'
            content = re.sub(pattern, replacement, content)

            # MergedBondInfo<I> in function signatures (after fn or in type position)
            pattern = rf'(Result<[^>]*{struct_name}<)I(>[^>]*>)'
            replacement = rf'\1T::Index\2'
            content = re.sub(pattern, replacement, content)

            # &MergedBondInfo<I> or HashMap<..., &MergedBondInfo<I>>
            pattern = rf'(&{struct_name}<)I(>)'
            replacement = rf'\1T::Index\2'
            content = re.sub(pattern, replacement, content)

        # Vec<I> -> Vec<T::Index> (but not in struct definitions)
        pattern = r'(?<![a-zA-Z_])Vec<I>(?![a-zA-Z_])'
        replacement = r'Vec<T::Index>'
        content = re.sub(pattern, replacement, content)

        # HashSet<I> -> HashSet<T::Index>
        pattern = r'(?<![a-zA-Z_])HashSet<I>(?![a-zA-Z_])'
        replacement = r'HashSet<T::Index>'
        content = re.sub(pattern, replacement, content)

        # HashMap<I, -> HashMap<T::Index,
        pattern = r'HashMap<I,'
        replacement = r'HashMap<T::Index,'
        content = re.sub(pattern, replacement, content)

        # &I -> &T::Index (in function params, not in closure |&i| patterns)
        pattern = r':\s*&I(?=\s*[,\)])'
        replacement = r': &T::Index'
        content = re.sub(pattern, replacement, content)

        # Option<&I> -> Option<&T::Index>
        pattern = r'Option<&I>'
        replacement = r'Option<&T::Index>'
        content = re.sub(pattern, replacement, content)

        # : I, or : I) -> : T::Index, or : T::Index)
        pattern = r':\s*I(?=\s*[,\)])'
        replacement = r': T::Index'
        content = re.sub(pattern, replacement, content)

        return content

    def fix_id_access(self, content: str) -> str:
        """Replace I::Id with <T::Index as IndexLike>::Id in TreeTN contexts."""
        if 'impl<T, V> TreeTN<T, V>' not in content:
            return content

        # I::Id -> <T::Index as IndexLike>::Id
        pattern = r'I::Id'
        replacement = r'<T::Index as IndexLike>::Id'
        content = re.sub(pattern, replacement, content)

        return content

    def fix_tensor_indices_access(self, content: str) -> str:
        """Replace tensor.indices with tensor.external_indices()."""
        # .indices.iter() -> .external_indices().iter()
        pattern = r'\.indices\.iter\(\)'
        replacement = r'.external_indices().iter()'
        content = re.sub(pattern, replacement, content)

        # .indices.len() -> .num_external_indices()
        pattern = r'\.indices\.len\(\)'
        replacement = r'.num_external_indices()'
        content = re.sub(pattern, replacement, content)

        # &tensor.indices -> tensor.external_indices()
        pattern = r'&(\w+)\.indices(?!\w)'
        replacement = r'\1.external_indices()'
        content = re.sub(pattern, replacement, content)

        # for idx in tensor.indices -> for idx in tensor.external_indices()
        pattern = r'for\s+(\w+)\s+in\s+(\w+)\.indices(?!\w)'
        replacement = r'for \1 in \2.external_indices()'
        content = re.sub(pattern, replacement, content)

        # tensor.indices.clone() -> tensor.external_indices()
        pattern = r'(\w+)\.indices\.clone\(\)'
        replacement = r'\1.external_indices()'
        content = re.sub(pattern, replacement, content)

        return content

    def fix_index_id_field_access(self, content: str) -> str:
        """Replace index.id (field) with index.id() (method)."""
        # .id == -> .id() ==
        pattern = r'\.id\s*=='
        replacement = r'.id() =='
        content = re.sub(pattern, replacement, content)

        # .id != -> .id() !=
        pattern = r'\.id\s*!='
        replacement = r'.id() !='
        content = re.sub(pattern, replacement, content)

        # .id.clone() -> .id().clone()
        pattern = r'\.id\.clone\(\)'
        replacement = r'.id().clone()'
        content = re.sub(pattern, replacement, content)

        # &idx.id -> idx.id() (when not followed by word char or open paren)
        pattern = r'&(\w+)\.id(?!\w|\()'
        replacement = r'\1.id()'
        content = re.sub(pattern, replacement, content)

        return content

    def remove_symm_and_tags_bounds(self, content: str) -> str:
        """Remove I::Symm, I::Tags, and related Symmetry trait bounds."""
        lines = content.split('\n')
        filtered_lines = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Skip lines that are I::Symm bounds
            if re.search(r'^\s*I::Symm\s*:', line):
                i += 1
                # Remove trailing comma from previous line if exists
                if filtered_lines and filtered_lines[-1].rstrip().endswith(','):
                    filtered_lines[-1] = re.sub(r',\s*$', '', filtered_lines[-1])
                continue

            # Skip lines that are <T::Index as IndexLike>::Symm bounds
            if re.search(r'^\s*<T::Index as IndexLike>::Symm\s*:', line):
                i += 1
                if filtered_lines and filtered_lines[-1].rstrip().endswith(','):
                    filtered_lines[-1] = re.sub(r',\s*$', '', filtered_lines[-1])
                continue

            # Skip lines that are I::Tags bounds
            if re.search(r'^\s*I::Tags\s*:', line):
                i += 1
                if filtered_lines and filtered_lines[-1].rstrip().endswith(','):
                    filtered_lines[-1] = re.sub(r',\s*$', '', filtered_lines[-1])
                continue

            # Remove inline I::Symm: From<NoSymmSpace>, patterns
            line = re.sub(r'\s*I::Symm:\s*From<NoSymmSpace>\s*,?\s*', '', line)

            # Remove inline I::Tags: Default, patterns
            line = re.sub(r'\s*I::Tags:\s*Default\s*,?\s*', '', line)

            filtered_lines.append(line)
            i += 1

        return '\n'.join(filtered_lines)

    def fix_remaining_i_references(self, content: str) -> str:
        """Fix remaining I:: references that should be T::Index based."""
        if 'impl<T, V> TreeTN<T, V>' not in content:
            return content

        # I::Symm remaining -> remove or replace depending on context
        # For now, mark as TODO for manual review
        if 'I::Symm' in content:
            # Remove I::Symm from type parameters like Index<I::Id, I::Symm, I::Tags>
            # This is complex - mark for manual fix
            pass

        # I::Tags remaining
        if 'I::Tags' in content:
            pass

        return content

    def fix_imports(self, content: str) -> str:
        """Fix import statements."""
        # Add TensorLike to imports if not present and needed
        if 'TensorLike' not in content and 'impl<T, V> TreeTN' in content:
            # Find tensor4all_core imports and add TensorLike
            pattern = r'use tensor4all_core::\{([^}]+)\};'
            def add_tensorlike(m):
                imports = m.group(1)
                if 'TensorLike' not in imports:
                    imports = imports.rstrip() + ', TensorLike'
                return f'use tensor4all_core::{{{imports}}};'
            content = re.sub(pattern, add_tensorlike, content)

        # Remove Symmetry import if no longer used
        if 'Symmetry' in content and 'I::Symm' not in content and ': Symmetry' not in content:
            # Check if Symmetry is actually used
            uses = re.findall(r'\bSymmetry\b', content)
            import_uses = re.findall(r'use.*Symmetry', content)
            if len(uses) == len(import_uses):
                # Only import usage, remove it
                content = re.sub(r',\s*Symmetry\b', '', content)
                content = re.sub(r'\bSymmetry\s*,\s*', '', content)

        return content

    def fix_where_clauses(self, content: str) -> str:
        """Clean up where clauses after other transformations."""
        # Remove empty where clauses
        pattern = r'where\s*\{'
        replacement = r'{'
        content = re.sub(pattern, replacement, content)

        # Fix double commas
        pattern = r',\s*,'
        replacement = r','
        content = re.sub(pattern, replacement, content)

        # Fix trailing commas before {
        pattern = r',\s*\n\s*\{'
        replacement = r'\n{'
        content = re.sub(pattern, replacement, content)

        # Fix comma followed by newline and { on same line
        pattern = r',\s*\{'
        replacement = r' {'
        content = re.sub(pattern, replacement, content)

        return content


def main():
    if len(sys.argv) < 2:
        print("Usage: python refactor_treetn.py <file_or_directory> [--dry-run]")
        print("\nRefactors treetn modules from TreeTN<I, V> to TreeTN<T, V> pattern.")
        print("\nNOTE: Helper structs (like MergedBondInfo<I>) that only hold indices")
        print("      should keep their I: IndexLike pattern - review manually.")
        sys.exit(1)

    path = Path(sys.argv[1])
    dry_run = '--dry-run' in sys.argv

    refactorer = RustRefactorer()

    if path.is_file():
        files = [path]
    else:
        files = list(path.rglob('*.rs'))

    for filepath in files:
        print(f"Processing: {filepath}")

        try:
            new_content = refactorer.refactor_file(filepath)
            original = filepath.read_text()

            if new_content != original:
                if dry_run:
                    print(f"  Would modify: {filepath}")
                    # Show diff preview
                    import difflib
                    diff = difflib.unified_diff(
                        original.splitlines(keepends=True),
                        new_content.splitlines(keepends=True),
                        fromfile=str(filepath),
                        tofile=str(filepath) + '.new'
                    )
                    for line in list(diff)[:80]:  # Show first 80 lines of diff
                        print(line, end='')
                    print("...")
                else:
                    filepath.write_text(new_content)
                    print(f"  Modified: {filepath}")
            else:
                print(f"  No changes: {filepath}")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
