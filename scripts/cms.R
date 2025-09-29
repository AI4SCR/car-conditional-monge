library(SingleCellExperiment)
library(CellMixS)

expression = read.table("/Users/alicedriessen/Box/LegacyFromOldColleagues/Alice/CAR_Tcells/Model/conditional-monge/experiments/cmonge/sel_CARs_LN_True_grad_acc_4_cosine/expression.csv", sep=",", header=TRUE, row.names=1)
umap_data = read.table("/Users/alicedriessen/Box/LegacyFromOldColleagues/Alice/CAR_Tcells/Model/conditional-monge/experiments/cmonge/sel_CARs_LN_True_grad_acc_4_cosine/umap_data.csv", sep=",", header=TRUE, row.names=1)
meta = read.table("/Users/alicedriessen/Box/LegacyFromOldColleagues/Alice/CAR_Tcells/Model/conditional-monge/experiments/cmonge/sel_CARs_LN_True_grad_acc_4_cosine/meta.csv", sep=",", header=TRUE, row.names=1)

subset_car <- expression$subset_CAR
avg_cms <- c()
for (x in unique(subset_car)){
    print(x)
    sel_expr <- expression[(expression$subset_CAR==x)&(meta$dtype %in% c("target", "transport")), !(names(expression)%in% c("subset_CAR"))]
    gene_names <- names(sel_expr)
    sel_umap <- umap_data[(umap_data$subset_CAR==x)&(meta$dtype %in% c("target", "transport")),!(names(umap_data)%in% c("subset_CAR"))]
    sel_meta <- meta[(meta$subset_CAR==x)&(meta$dtype %in% c("target", "transport")),]
    sel_expr <- t(as.matrix(sel_expr))

    print(dim(sel_expr))

    sce <- SingleCellExperiment(list(logcounts=sel_expr), colData=sel_meta, rowData=DataFrame(name=gene_names))
    reducedDims(sce) <- list(UMAP=sel_umap)
    sce <- cms(sce, k = 70, group = "dtype", assay_name = "logcounts")
    coldata <- colData(sce)
    cms <- mean(coldata$cms)
    avg_cms <- c(avg_cms, cms)
}

res <- data.frame(subset_CAR=unique(subset_car), avg_cms=avg_cms)
write.csv(res, "/Users/alicedriessen/Box/LegacyFromOldColleagues/Alice/CAR_Tcells/Model/conditional-monge/experiments/cmonge/sel_CARs_LN_True_grad_acc_4_cosine/cms_scores.csv", row.names=FALSE)
print(res)
# expression <- expression[,!(names(expression)%in% c("subset_CAR"))]
# gene_names <- names(expression)
# expression <- t(expression)
# umap_data <- umap_data[,!(names(umap_data)%in% c("subset_CAR"))]

# expr <- as.matrix(expression) 

# head(expression)
# head(umap_data)
# head(meta)
# head(subset_car)
# head(expr)

# sce <- SingleCellExperiment(list(logcounts=expression), colData=meta, rowData=DataFrame(name=gene_names))

# reducedDims(sce) <- list(UMAP=umap_data)
# head(reducedDim(sce, "UMAP")[,1:2])
# sce_cms <- cms(sce, k = 70, group = "dtype", assay_name = "logcounts")
# sce_cms