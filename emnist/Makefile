.PHONY: upload download

upload: metacentrum/upload_list.txt
	rsync -urv --no-relative --files-from metacentrum/upload_list.txt ./ metacentrum:emnist/

download:
	rsync -urv metacentrum:emnist/models/ ./
