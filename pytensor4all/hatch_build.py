"""Custom build hook for hatchling to create platform-specific wheels."""

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Build hook that enables platform-specific wheel tags."""

    def initialize(self, version, build_data):
        """Set infer_tag to True to generate platform-specific wheels."""
        # This makes hatchling detect that we have native code
        # and generate appropriate wheel tags (e.g., cp311-cp311-macosx_arm64)
        build_data["infer_tag"] = True
