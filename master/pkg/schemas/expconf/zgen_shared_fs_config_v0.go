// Code generated by gen.py. DO NOT EDIT.

package expconf

import (
	"github.com/santhosh-tekuri/jsonschema/v2"

	"github.com/determined-ai/determined/master/pkg/schemas"
)

func (s SharedFSConfigV0) HostPath() string {
	if s.RawHostPath == nil {
		panic("You must call WithDefaults on SharedFSConfigV0 before .HostPath")
	}
	return *s.RawHostPath
}

func (s *SharedFSConfigV0) SetHostPath(val string) {
	s.RawHostPath = &val
}

func (s SharedFSConfigV0) ContainerPath() *string {
	return s.RawContainerPath
}

func (s *SharedFSConfigV0) SetContainerPath(val *string) {
	s.RawContainerPath = val
}

func (s SharedFSConfigV0) CheckpointPath() *string {
	return s.RawCheckpointPath
}

func (s *SharedFSConfigV0) SetCheckpointPath(val *string) {
	s.RawCheckpointPath = val
}

func (s SharedFSConfigV0) TensorboardPath() *string {
	return s.RawTensorboardPath
}

func (s *SharedFSConfigV0) SetTensorboardPath(val *string) {
	s.RawTensorboardPath = val
}

func (s SharedFSConfigV0) StoragePath() *string {
	return s.RawStoragePath
}

func (s *SharedFSConfigV0) SetStoragePath(val *string) {
	s.RawStoragePath = val
}

func (s SharedFSConfigV0) Propagation() string {
	if s.RawPropagation == nil {
		panic("You must call WithDefaults on SharedFSConfigV0 before .Propagation")
	}
	return *s.RawPropagation
}

func (s *SharedFSConfigV0) SetPropagation(val string) {
	s.RawPropagation = &val
}

func (s SharedFSConfigV0) ParsedSchema() interface{} {
	return schemas.ParsedSharedFSConfigV0()
}

func (s SharedFSConfigV0) SanityValidator() *jsonschema.Schema {
	return schemas.GetSanityValidator("http://determined.ai/schemas/expconf/v0/shared-fs.json")
}

func (s SharedFSConfigV0) CompletenessValidator() *jsonschema.Schema {
	return schemas.GetCompletenessValidator("http://determined.ai/schemas/expconf/v0/shared-fs.json")
}
