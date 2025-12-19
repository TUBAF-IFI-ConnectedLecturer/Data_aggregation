# Shared Resources

This directory contains resources shared across multiple pipelines.

## Files

### `dewey_classification.txt`

Dewey Decimal Classification (DDC) reference file used by all pipelines for document classification.

**Format**: Text file with DDC notations and labels in German

**Usage**:
- Referenced by `AIMetaDataExtraction` stage
- Used for two-step Dewey classification:
  1. Thematic analysis (identify research fields)
  2. Dewey mapping (map to official DDC notations)

**Pipelines using this resource**:
- `local_pdfs` - Scientific paper classification
- `opal` - Educational material classification

## Adding New Shared Resources

When adding new shared resources:

1. Place file in this directory
2. Update this README with description
3. Reference in pipeline configs with relative path:
   ```yaml
   resource_file: ../../shared/resource_name.ext
   ```

## Notes

- Shared resources should be version-controlled
- Breaking changes to shared resources affect all pipelines
- Test changes with both pipelines before committing
