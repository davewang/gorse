// Copyright 2020 gorse Project Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package config

import (
	"testing"

	"github.com/BurntSushi/toml"
	"github.com/stretchr/testify/assert"
)

func TestLoadConfig(t *testing.T) {
	config, _, err := LoadConfig("../misc/config_test/config.toml")
	assert.Nil(t, err)

	// database configuration
	assert.Equal(t, "redis://localhost:6379", config.Database.CacheStore)
	assert.Equal(t, "mysql://root@tcp(localhost:3306)/gorse?parseTime=true", config.Database.DataStore)
	assert.Equal(t, true, config.Database.AutoInsertUser)
	assert.Equal(t, false, config.Database.AutoInsertItem)
	assert.Equal(t, []string{"star", "fork"}, config.Database.PositiveFeedbackType)
	assert.Equal(t, uint(998), config.Database.PositiveFeedbackTTL)
	assert.Equal(t, uint(999), config.Database.ItemTTL)

	// master configuration
	assert.Equal(t, 8086, config.Master.Port)
	assert.Equal(t, "127.0.0.1", config.Master.Host)
	assert.Equal(t, 8088, config.Master.HttpPort)
	assert.Equal(t, "127.0.0.1", config.Master.HttpHost)
	assert.Equal(t, 3, config.Master.SearchJobs)
	assert.Equal(t, 4, config.Master.FitJobs)
	assert.Equal(t, 30, config.Master.MetaTimeout)

	// server configuration
	assert.Equal(t, 128, config.Server.DefaultN)
	assert.Equal(t, "p@ssword", config.Server.APIKey)

	// recommend configuration
	assert.Equal(t, 12, config.Recommend.PopularWindow)
	assert.Equal(t, 66, config.Recommend.FitPeriod)
	assert.Equal(t, 88, config.Recommend.SearchPeriod)
	assert.Equal(t, 102, config.Recommend.SearchEpoch)
	assert.Equal(t, 9, config.Recommend.SearchTrials)
	assert.Equal(t, "latest", config.Recommend.FallbackRecommend)
}

func TestConfig_FillDefault(t *testing.T) {
	var config Config
	meta, err := toml.Decode("", &config)
	assert.Nil(t, err)
	config.FillDefault(meta)
	assert.Equal(t, *(*Config)(nil).LoadDefaultIfNil(), config)
}
