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

package master

import (
	"fmt"
	"github.com/emicklei/go-restful/v3"
	"github.com/zhenghaoz/gorse/model"
	"github.com/zhenghaoz/gorse/server"
	"go.uber.org/zap"
	"math/rand"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"

	"github.com/ReneKroon/ttlcache/v2"
	"github.com/zhenghaoz/gorse/base"
	"github.com/zhenghaoz/gorse/config"
	"github.com/zhenghaoz/gorse/model/pr"
	"github.com/zhenghaoz/gorse/protocol"
	"github.com/zhenghaoz/gorse/storage/cache"
	"github.com/zhenghaoz/gorse/storage/data"
	"google.golang.org/grpc"
)

const (
	ServerNode = "Server"
	WorkerNode = "Worker"
)

type Master struct {
	protocol.UnimplementedMasterServer
	server.RestServer

	// cluster meta cache
	ttlCache       *ttlcache.Cache
	nodesInfo      map[string]*Node
	nodesInfoMutex sync.Mutex

	// users index
	userIndex        base.Index
	userIndexVersion int64
	userIndexMutex   sync.Mutex

	// personal ranking model
	prModel     pr.Model
	prModelName string
	prVersion   int64
	prScore     pr.Score
	prMutex     sync.Mutex
	prSearcher  *pr.ModelSearcher

	// factorization machine
	//fmModel    ctr.FactorizationMachine
	//ctrVersion int64
	//fmMutex    sync.mutex

	localCache *LocalCache
}

func NewMaster(cfg *config.Config) *Master {
	rand.Seed(time.Now().UnixNano())
	return &Master{
		nodesInfo: make(map[string]*Node),
		// init versions
		prVersion: rand.Int63(),
		// ctrVersion:       rand.Int63(),
		userIndexVersion: rand.Int63(),
		// default model
		prModelName: "bpr",
		prModel:     pr.NewBPR(nil),
		prSearcher:  pr.NewModelSearcher(cfg.Recommend.SearchEpoch, cfg.Recommend.SearchTrials),
		RestServer: server.RestServer{
			GorseConfig: cfg,
			HttpHost:    cfg.Master.HttpHost,
			HttpPort:    cfg.Master.HttpPort,
			EnableAuth:  false,
			WebService:  new(restful.WebService),
		},
	}
}

func (m *Master) Serve() {

	// load local cached model
	var err error
	m.localCache, err = LoadLocalCache(filepath.Join(os.TempDir(), "gorse-master"))
	if err != nil {
		base.Logger().Error("failed to load local cache", zap.Error(err))
	}
	if m.localCache.Model != nil {
		base.Logger().Info("load cached model",
			zap.String("model_name", m.localCache.ModelName),
			zap.String("model_version", base.Hex(m.localCache.ModelVersion)),
			zap.Float32("model_score", m.localCache.ModelScore.NDCG),
			zap.Any("params", m.localCache.Model.GetParams()))
		m.prModel = m.localCache.Model
		m.prModelName = m.localCache.ModelName
		m.prVersion = m.localCache.ModelVersion
		m.prScore = m.localCache.ModelScore
	}

	// create cluster meta cache
	m.ttlCache = ttlcache.NewCache()
	m.ttlCache.SetExpirationCallback(m.nodeDown)
	m.ttlCache.SetNewItemCallback(m.nodeUp)
	if err = m.ttlCache.SetTTL(
		time.Duration(m.GorseConfig.Master.MetaTimeout+10) * time.Second,
	); err != nil {
		base.Logger().Fatal("failed to set TTL", zap.Error(err))
	}

	// connect data database
	m.DataStore, err = data.Open(m.GorseConfig.Database.DataStore)
	if err != nil {
		base.Logger().Fatal("failed to connect data database", zap.Error(err))
	}
	if err = m.DataStore.Init(); err != nil {
		base.Logger().Fatal("failed to init database", zap.Error(err))
	}

	// connect cache database
	m.CacheStore, err = cache.Open(m.GorseConfig.Database.CacheStore)
	if err != nil {
		base.Logger().Fatal("failed to connect cache database", zap.Error(err),
			zap.String("database", m.GorseConfig.Database.CacheStore))
	}

	go m.StartHttpServer()
	go m.FitLoop()
	base.Logger().Info("start model fit", zap.Int("period", m.GorseConfig.Recommend.FitPeriod))
	go m.SearchLoop()
	base.Logger().Info("start model searcher", zap.Int("period", m.GorseConfig.Recommend.SearchPeriod))

	// start rpc server
	base.Logger().Info("start rpc server",
		zap.String("host", m.GorseConfig.Master.Host),
		zap.Int("port", m.GorseConfig.Master.Port))
	lis, err := net.Listen("tcp", fmt.Sprintf("%s:%d", m.GorseConfig.Master.Host, m.GorseConfig.Master.Port))
	if err != nil {
		base.Logger().Fatal("failed to listen", zap.Error(err))
	}
	var opts []grpc.ServerOption
	grpcServer := grpc.NewServer(opts...)
	protocol.RegisterMasterServer(grpcServer, m)
	if err = grpcServer.Serve(lis); err != nil {
		base.Logger().Fatal("failed to start rpc server", zap.Error(err))
	}
}

func (m *Master) FitLoop() {
	defer base.CheckPanic()
	lastNumUsers, lastNumItems, lastNumFeedback := 0, 0, 0
	var bestName string
	var bestModel pr.Model
	var bestScore pr.Score
	for {
		// download dataset
		base.Logger().Info("load dataset for model fit", zap.Strings("feedback_types", m.GorseConfig.Database.PositiveFeedbackType))
		dataSet, items, feedbacks, err := pr.LoadDataFromDatabase(m.DataStore, m.GorseConfig.Database.PositiveFeedbackType,
			m.GorseConfig.Database.ItemTTL, m.GorseConfig.Database.PositiveFeedbackTTL)
		if err != nil {
			base.Logger().Error("failed to load database", zap.Error(err))
			goto sleep
		}
		// save stats
		if err = m.CacheStore.SetString(cache.GlobalMeta, cache.NumUsers, strconv.Itoa(dataSet.UserCount())); err != nil {
			base.Logger().Error("failed to write meta", zap.Error(err))
		}
		if err = m.CacheStore.SetString(cache.GlobalMeta, cache.NumItems, strconv.Itoa(dataSet.ItemCount())); err != nil {
			base.Logger().Error("failed to write meta", zap.Error(err))
		}
		if err = m.CacheStore.SetString(cache.GlobalMeta, cache.NumPositiveFeedback, strconv.Itoa(dataSet.Count())); err != nil {
			base.Logger().Error("failed to write meta", zap.Error(err))
		}
		// sleep if empty
		if dataSet.Count() == 0 {
			base.Logger().Warn("empty dataset", zap.Strings("feedback_type", m.GorseConfig.Database.PositiveFeedbackType))
			goto sleep
		}
		// check best model
		bestName, bestModel, bestScore = m.prSearcher.GetBestModel()
		m.prMutex.Lock()
		if bestName != "" &&
			(bestName != m.prModelName || bestModel.GetParams().ToString() != m.prModel.GetParams().ToString()) &&
			(bestScore.NDCG > m.prScore.NDCG) {
			// 1. best model must have been found.
			// 2. best model must be different from current model
			// 3. best model must perform better than current model
			m.prModel = bestModel
			m.prModelName = bestName
			base.Logger().Info("find better model",
				zap.String("name", bestName),
				zap.Any("params", m.prModel.GetParams()))
		} else if dataSet.UserCount() == lastNumUsers && dataSet.ItemCount() == lastNumItems && dataSet.Count() == lastNumFeedback {
			// sleep if nothing changed
			m.prMutex.Unlock()
			goto sleep
		}
		m.prMutex.Unlock()
		lastNumUsers, lastNumItems, lastNumFeedback = dataSet.UserCount(), dataSet.ItemCount(), dataSet.Count()
		// update user index
		m.userIndexMutex.Lock()
		m.userIndex = dataSet.UserIndex
		m.userIndexVersion++
		m.userIndexMutex.Unlock()
		// fit model
		m.fitPRModel(dataSet, m.prModel)
		// collect similar items
		m.similar(items, dataSet, model.SimilarityDot)
		// collect popular items
		m.popItem(items, feedbacks)
		// collect latest items
		m.latest(items)
		// sleep
	sleep:
		time.Sleep(time.Duration(m.GorseConfig.Recommend.FitPeriod) * time.Minute)
	}
}

// SearchLoop searches optimal recommendation model in background. It never modifies variables other than prSearcher.
func (m *Master) SearchLoop() {
	defer base.CheckPanic()
	lastNumUsers, lastNumItems, lastNumFeedback := 0, 0, 0
	for {
		var trainSet, valSet *pr.DataSet
		// download dataset
		base.Logger().Info("load dataset for model search", zap.Strings("feedback_types", m.GorseConfig.Database.PositiveFeedbackType))
		dataSet, _, _, err := pr.LoadDataFromDatabase(m.DataStore, m.GorseConfig.Database.PositiveFeedbackType,
			m.GorseConfig.Database.ItemTTL, m.GorseConfig.Database.PositiveFeedbackTTL)
		if err != nil {
			base.Logger().Error("failed to load database", zap.Error(err))
			goto sleep
		}
		// sleep if empty
		if dataSet.Count() == 0 {
			base.Logger().Warn("empty dataset", zap.Strings("feedback_type", m.GorseConfig.Database.PositiveFeedbackType))
			goto sleep
		}
		// sleep if nothing changed
		if dataSet.UserCount() == lastNumUsers && dataSet.ItemCount() == lastNumItems && dataSet.Count() == lastNumFeedback {
			goto sleep
		}
		// start search
		trainSet, valSet = dataSet.Split(0, 0)
		err = m.prSearcher.Fit(trainSet, valSet)
		if err != nil {
			base.Logger().Error("failed to search model", zap.Error(err))
		}
	sleep:
		time.Sleep(time.Duration(m.GorseConfig.Recommend.SearchPeriod) * time.Minute)
	}
}
