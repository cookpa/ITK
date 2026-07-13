/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#include "gtest/gtest.h"
#include "itkGiplImageIO.h"
#include "itkImage.h"
#include "itkImageFileWriter.h"

#include <fstream>
#include <sstream>

#define _STRING(s) #s
#define TOSTRING(s) std::string(_STRING(s))

namespace
{
std::string
OutputPath(const std::string & name)
{
  return TOSTRING(ITK_TEST_OUTPUT_DIR) + "/" + name;
}
} // namespace

TEST(GiplImageIO, WriteOfUnsupportedComponentTypePreservesExistingFile)
{
  const std::string path = OutputPath("gipl_b53_existing.gipl");
  const std::string originalContents = "pre-existing data that must survive a failed Gipl write";
  {
    std::ofstream out(path, std::ios::binary);
    out << originalContents;
  }

  using UnsupportedImageType = itk::Image<unsigned long, 2>;
  auto                                 image = UnsupportedImageType::New();
  const UnsupportedImageType::SizeType size{ { 2, 2 } };
  image->SetRegions(UnsupportedImageType::RegionType(size));
  image->Allocate();
  image->FillBuffer(7);

  auto giplIO = itk::GiplImageIO::New();
  using WriterType = itk::ImageFileWriter<UnsupportedImageType>;
  auto writer = WriterType::New();
  writer->SetImageIO(giplIO);
  writer->SetInput(image);
  writer->SetFileName(path);

  EXPECT_THROW(writer->Update(), itk::ExceptionObject);

  std::ifstream      in(path, std::ios::binary);
  std::ostringstream contents;
  contents << in.rdbuf();
  EXPECT_EQ(contents.str(), originalContents);
}
