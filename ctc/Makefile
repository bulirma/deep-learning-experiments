.PHONY: upload download

upload: gpulab/upload_list.txt
	rsync -urv --no-relative --files-from gpulab/upload_list.txt ./ gpulab:ctc/

download:
	rsync -urv gpulab:ctc/models/ ./models/
