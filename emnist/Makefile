.PHONY: upload download localupload localdownload

upload: metacentrum/upload_list.txt
	rsync -urv --no-relative --files-from metacentrum/upload_list.txt ./ metacentrum:emnist/

download:
	rsync -urv metacentrum:emnist/models/ ./models/

localupload:
	rsync -urv --no-relative --files-from localpower/upload_list.txt ./ localpower:projects/emnist/

localdownload:
	rsync -urv localpower:projects/emnist/models/ ./models/
