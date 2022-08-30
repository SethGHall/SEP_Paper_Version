# Set dir of Makefile to a variable to use later
MAKEPATH := $(abspath $(lastword $(MAKEFILE_LIST)))
BASEDIR := $(patsubst %/,%,$(dir $(MAKEPATH)))

# Cross platform make declarations
# \
!ifndef 0 # \
# nmake code here \
MKDIR=mkdir # \
RMRF=del /f /s /q # \
!else
# make code here
MKDIR=mkdir -p
RMRF=rm -rf
# \
!endif


# Args for K8s and charts
NAME := sep-imaging-pipeline
KUBE_NAMESPACE ?= "default"
KUBECTL_VERSION ?= latest
HELM_VERSION ?= v3.1.2
# See : https://quay.io/repository/helmpack/chart-testing?tab=tags
CT_TAG ?= v3.0.0-beta.2
HELM_CHART = $(NAME)
HELM_RELEASE ?= test-receive
CI_REGISTRY ?= docker.io
CI_REPOSITORY ?= localhost:5000
TAG ?= latest
IMAGE = $(CI_REPOSITORY)/$(NAME):$(TAG)
VALUES_FILE ?= charts/$(HELM_CHART)/values.yaml

BUILDDIR := $(BASEDIR)/build
RUNDIR := $(BASEDIR)/app
DATADIR := $(BASEDIR)/imaging_data
CONFIGDIR := $(BASEDIR)/imaging_configs
CHARTCONFIGDIR := $(BASEDIR)/charts/$(HELM_CHART)/imageconfigs
IMAGING_PIPELINE := imaging
RUNTIME ?= nvidia ## Docker runtime
BUILD_TYPE ?= Debug ## build type: Debug or Release
BUILD_TESTS ?= OFF ## cmake enable tests

# define your personal overides for above variables in here
-include PrivateRules.mak

.PHONY: vars help test k8s show lint deploy delete logs describe namespace default all clean
.DEFAULT_GOAL := help

# Set dir of Makefile to a variable to use later
MAKEPATH := $(abspath $(lastword $(MAKEFILE_LIST)))
BASEDIR := $(patsubst %/,%,$(dir $(MAKEPATH)))

reboot: ## Purge Docker, removing containers/images, and restarting with a fresh registry running on localhost:5000
	docker rmi $$(docker images -a -q) \
	&& docker stop $$(docker ps -a -q) \
	&& docker rm $$(docker ps -a -q) \
	&& sudo systemctl restart docker \
	&& docker run -d -p 5000:5000 --restart=always --name registry registry:2

image: ## build Docker image
	docker build -t $(NAME):latest -f Dockerfile .

tag: image ## tag image
	docker tag $(NAME):latest $(IMAGE)

push: tag ## push image
	docker push $(IMAGE)

test: ## test IMAGING_PIPELINE image
	docker run --rm --gpus all -ti -v $$(pwd)/imaging_data:/imaging_data $(IMAGE)

clean: ## Delete the build folder and compiled software for the imaging pipeline
	$(RMRF) $(BUILDDIR)

build: ## Build the imaging pipeline for local execution (no kubernetes)
	$(MKDIR) $(BUILDDIR)
	cd $(BUILDDIR) && cmake -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(BASEDIR)
	cd $(BUILDDIR) && $(MAKE) -j8

docs: ## Builds the Sphinx/ReadTheDocs documentation for imaging pipeline
	BUILD_TYPE ?= DOCS
	$(MAKE) build

vars: ## make variables
	@echo "VALUES_FILE: $(VALUES_FILE)"

k8s: vars ## Which kubernetes are we connected to
	@echo "Kubernetes cluster-info:"
	@kubectl cluster-info
	@echo ""
	@echo "kubectl version:"
	@kubectl version
	@echo ""
	@echo "Helm version:"
	@helm version --client
	@echo ""
	@echo "Helm plugins:"
	@helm plugin list

logs: ## show receiver POD logs
	@for i in `kubectl -n $(KUBE_NAMESPACE) get pods -l app.kubernetes.io/instance=$(HELM_RELEASE) -o=name`; \
	do \
	echo "---------------------------------------------------"; \
	echo "Logs for $${i}"; \
	echo kubectl -n $(KUBE_NAMESPACE) logs $${i}; \
	echo kubectl -n $(KUBE_NAMESPACE) get $${i} -o jsonpath="{.spec.initContainers[*].name}"; \
	echo "---------------------------------------------------"; \
	for j in `kubectl -n $(KUBE_NAMESPACE) get $${i} -o jsonpath="{.spec.initContainers[*].name}"`; do \
	RES=`kubectl -n $(KUBE_NAMESPACE) logs $${i} -c $${j} 2>/dev/null`; \
	echo "initContainer: $${j}"; echo "$${RES}"; \
	echo "---------------------------------------------------";\
	done; \
	echo "Main Pod logs for $${i}"; \
	echo "---------------------------------------------------"; \
	for j in `kubectl -n $(KUBE_NAMESPACE) get $${i} -o jsonpath="{.spec.containers[*].name}"`; do \
	RES=`kubectl -n $(KUBE_NAMESPACE) logs $${i} -c $${j} 2>/dev/null`; \
	echo "Container: $${j}"; echo "$${RES}"; \
	echo "---------------------------------------------------";\
	done; \
	echo "---------------------------------------------------"; \
	echo ""; echo ""; echo ""; \
	done

redeploy: delete deploy  ## redeploy sep-pipeline-imaging

namespace: ## create the kubernetes namespace
	kubectl describe namespace $(KUBE_NAMESPACE) || kubectl create namespace $(KUBE_NAMESPACE)

delete_namespace: ## delete the kubernetes namespace
	@if [ "default" == "$(KUBE_NAMESPACE)" ] || [ "kube-system" == "$(KUBE_NAMESPACE)" ]; then \
	echo "You cannot delete Namespace: $(KUBE_NAMESPACE)"; \
	exit 1; \
	else \
	kubectl describe namespace $(KUBE_NAMESPACE) && kubectl delete namespace $(KUBE_NAMESPACE); \
	fi

deploy: install

install: namespace  ## install the helm chart
	cp -fa $(CONFIGDIR)/. $(CHARTCONFIGDIR)/
	helm install $(HELM_RELEASE) charts/$(HELM_CHART)/ \
				--wait \
				--namespace $(KUBE_NAMESPACE) \
				--set testDataPath=$(BASEDIR)/ \
				 --values $(VALUES_FILE)

delete: ## delete the helm chart release
	@helm delete $(HELM_RELEASE) --namespace $(KUBE_NAMESPACE)

show: vars ## show the helm chart
	helm template $(HELM_RELEASE) charts/$(HELM_CHART)/ \
				 --namespace $(KUBE_NAMESPACE) \
				 --set testDataPath=$(BASEDIR)/ \
				 --values $(VALUES_FILE)

lint: vars ## lint check the helm chart
	# Chart testing: https://github.com/helm/chart-testing
	@helm lint charts/$(HELM_CHART)/ \
				 --namespace $(KUBE_NAMESPACE) \
				 --set testDataPath=$(BASEDIR)/ \
				 --values $(VALUES_FILE)
	@docker run --rm \
	  --volume $(BASEDIR):/app \
	  quay.io/helmpack/chart-testing:$(CT_TAG) \
	  sh -c 'cd /app; ct lint --config ci/ct.yaml --all'

describe: ## describe Pods executed from Helm chart
	@for i in `kubectl -n $(KUBE_NAMESPACE) get pods -l app.kubernetes.io/instance=$(HELM_RELEASE) -o=name`; \
	do echo "---------------------------------------------------"; \
	echo "Describe for $${i}"; \
	echo kubectl -n $(KUBE_NAMESPACE) describe $${i}; \
	echo "---------------------------------------------------"; \
	kubectl -n $(KUBE_NAMESPACE) describe $${i}; \
	echo "---------------------------------------------------"; \
	echo ""; echo ""; echo ""; \
	done

helm_dependencies: ## Utility target to install Helm dependencies
	@which helm ; rc=$$?; \
	if [ $$rc != 0 ]; then \
	curl "https://get.helm.sh/helm-$(HELM_VERSION)-linux-amd64.tar.gz" | tar zx; \
	sudo mv -f linux-amd64/helm /usr/local/bin/; \
	rm -rf linux-amd64; \
	else \
	helm version | grep $(HELM_VERSION); rc=$$?; \
	if  [ $$rc != 0 ]; then \
	curl "https://get.helm.sh/helm-$(HELM_VERSION)-linux-amd64.tar.gz" | tar zx; \
	sudo mv -f linux-amd64/helm /usr/local/bin/; \
	rm -rf linux-amd64; \
	fi; \
	fi

kubectl_dependencies: ## Utility target to install K8s dependencies
	@which kubectl ; rc=$$?; \
	if [ $$rc != 0 ]; then \
		sudo curl -L -o /usr/bin/kubectl "https://storage.googleapis.com/kubernetes-release/release/$(KUBECTL_VERSION)/bin/linux/amd64/kubectl"; \
		sudo chmod +x /usr/bin/kubectl; \
	fi
	@printf "\nkubectl client version:"
	@kubectl version --client
	@printf "\nkubectl config view:"
	@kubectl config view
	@printf "\nkubectl config get-contexts:"
	@kubectl config get-contexts
	@printf "\nkubectl version:"
	@kubectl version

help:  ## show this help.
	@echo "$(MAKE) targets:"
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ": .*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
	@echo ""; echo "make vars (+defaults):"
	@grep -E '^[0-9a-zA-Z_-]+ \?=.*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = " \\?\\= | ## "}; {printf "\033[36m%-30s\033[0m %-20s %-30s\n", $$1, $$2, $$3}'

